/**
 * @file lpl_kmod.c
 * @brief Laplace Kernel Module — High-Performance Zero-Copy RingBuffer RX/TX.
 *
 * Character device /dev/lpl0 with:
 *   - Netfilter PRE_ROUTING hook to capture UDP packets before the stack
 *   - vmalloc_user + mmap for zero-copy shared memory (RX/TX ring buffers)
 *   - Lockless SPSC ring buffers via smp_load_acquire/smp_store_release
 *   - TX kthread with wait_event_interruptible + ioctl kick
 *   - Stats reporting via ioctl
 *
 * Build: out-of-tree via Kbuild.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/in.h>
#include <linux/inet.h>
#include <linux/init.h>
#include <linux/ip.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/net.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/skbuff.h>
#include <linux/socket.h>
#include <linux/uaccess.h>
#include <linux/udp.h>
#include <linux/vmalloc.h>
#include <linux/wait.h>

#include "lpl_protocol.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("MasterLaplace");
MODULE_DESCRIPTION("Laplace Kernel Module: High-Perf Zero-Copy RingBuffer RX/TX");
MODULE_VERSION("0.2");

/* ─── Module state ──────────────────────────────────────────────────────── */

static dev_t lpl_devno;
static struct cdev lpl_cdev;
static struct class *lpl_class;
static struct device *lpl_device;

static LplSharedMemory *shm; /* vmalloc_user shared memory    */

/**
 * @brief Serializes the RX producers.
 *
 * @details The Netfilter hook runs on whichever CPU received the packet, so with several receive
 *          queues, RPS or loopback traffic from several CPUs, two CPUs claim a slot at once. The
 *          lock makes the hook the single producer the ring is built for, and guards the RX counters.
 */
static DEFINE_SPINLOCK(rx_lock);

static uint64_t rx_packets; /**< Under rx_lock. */
static uint64_t rx_bytes;   /**< Under rx_lock. */
static uint64_t rx_drops;   /**< Under rx_lock. */
static uint64_t tx_packets; /**< Written by the TX thread only. */
static uint64_t tx_bytes;   /**< Written by the TX thread only. */
static uint64_t tx_drops;   /**< Written by the TX thread only. */

static uint32_t rx_write_index;       /**< Next RX slot to fill, under rx_lock; published to shm->rx.writer. */
static uint32_t rx_cached_read_index; /**< Last read index of the process the hook read, under rx_lock. */

/** @brief Serializes the TX thread's runs with LPL_IOCTL_RESET and LPL_IOCTL_GET_STATS. */
static DEFINE_MUTEX(tx_lock);

static uint32_t tx_read_index;         /**< Next TX slot to send, under tx_lock; published to shm->tx.reader. */
static uint32_t tx_cached_write_index; /**< Last write index of the process the thread read, under tx_lock. */

/** Most TX slots sent before the thread publishes its read index and yields. */
#define LPL_TX_RUN 64U

static struct task_struct *tx_task; /* TX kthread                    */
static wait_queue_head_t tx_wq;     /* wait queue for TX kick        */

static struct socket *udp_sock; /* kernel-space UDP socket       */

static struct nf_hook_ops lpl_nf_ops; /* Netfilter hook registration   */

/* ─── UDP send from kernel space ────────────────────────────────────────── */

/**
 * @brief Sends one TX slot from the kernel UDP socket.
 *
 * @details The slot lives in memory the process maps and may rewrite at any moment, so its header
 *          is read once, checked, and only the copies are used: a length checked in the slot and
 *          read again from it could have grown in between, and kernel_sendmsg would read past the
 *          slot.
 *
 * @param pkt The slot, in the shared mapping.
 * @return The bytes sent, or a negative errno.
 */
static int send_udp_packet(const LplTxPacket *pkt)
{
    struct msghdr msg = {};
    struct kvec iov;
    struct sockaddr_in dst;
    const uint16_t length = READ_ONCE(pkt->length);
    const uint16_t dst_port = READ_ONCE(pkt->dst_port);
    const uint32_t dst_ip = READ_ONCE(pkt->dst_ip);

    if (!udp_sock || length == 0 || length > LPL_MAX_PACKET_SIZE)
        return -EINVAL;

    memset(&dst, 0, sizeof(dst));
    dst.sin_family = AF_INET;
    dst.sin_port = htons(dst_port);
    dst.sin_addr.s_addr = htonl(dst_ip);

    iov.iov_base = (void *) pkt->data;
    iov.iov_len = length;

    msg.msg_name = &dst;
    msg.msg_namelen = sizeof(dst);

    return kernel_sendmsg(udp_sock, &msg, &iov, 1, length);
}

/* ─── TX kthread ────────────────────────────────────────────────────────── */

/**
 * @brief TX slots the process published and the thread has not sent, under tx_lock.
 *
 * @details Reads the process's write index again only when the cached copy shows none. An index
 *          that claims more slots than the ring holds was not written by a process that follows the
 *          protocol: the thread skips to it and counts one drop, rather than sending stale slots.
 */
static uint32_t tx_ready_slots(void)
{
    uint32_t ready = tx_cached_write_index - tx_read_index;

    if (ready == 0)
    {
        tx_cached_write_index = smp_load_acquire(&shm->tx.writer.write_index);
        ready = tx_cached_write_index - tx_read_index;
    }
    if (ready > LPL_RING_SLOTS)
    {
        tx_drops++;
        tx_read_index = tx_cached_write_index;
        smp_store_release(&shm->tx.reader.read_index, tx_read_index);
        return 0;
    }
    return ready;
}

/**
 * @brief Sends up to LPL_TX_RUN of the @p ready slots, then publishes the read index once, under tx_lock.
 */
static void tx_send_run(uint32_t ready)
{
    const uint32_t run = ready < LPL_TX_RUN ? ready : LPL_TX_RUN;
    uint32_t offset;

    for (offset = 0; offset < run; ++offset)
    {
        const int sent = send_udp_packet(&shm->tx.packets[(tx_read_index + offset) & LPL_RING_MASK]);

        if (sent >= 0)
        {
            tx_packets++;
            tx_bytes += (uint64_t) sent;
        }
        else
        {
            tx_drops++;
        }
    }
    tx_read_index += run;
    smp_store_release(&shm->tx.reader.read_index, tx_read_index);
}

/**
 * @brief Sleeps until the process publishes a slot past those sent, or the module unloads.
 *
 * @details The thread says it sleeps, then checks the ring a last time; the process publishes,
 *          then reads that word. With a full barrier on each side, either the thread sees the new
 *          slot or the process sees the thread asleep and kicks it, and a process that finds the
 *          thread awake publishes with no system call.
 */
static void tx_sleep_until_kicked(void)
{
    WRITE_ONCE(shm->tx.wake.sleeping, 1U);
    smp_mb();
    wait_event_interruptible(tx_wq, kthread_should_stop() ||
                                        smp_load_acquire(&shm->tx.writer.write_index) != READ_ONCE(tx_read_index));
    WRITE_ONCE(shm->tx.wake.sleeping, 0U);
}

/**
 * @brief Drains the TX ring a run at a time, and sleeps when it is empty.
 */
static int tx_thread_fn(void *data)
{
    (void) data;

    while (!kthread_should_stop())
    {
        uint32_t ready;

        mutex_lock(&tx_lock);
        ready = tx_ready_slots();
        if (ready != 0)
            tx_send_run(ready);
        mutex_unlock(&tx_lock);

        if (ready == 0)
            tx_sleep_until_kicked();
        else
            cond_resched();
    }

    return 0;
}

/* ─── Netfilter hook: capture UDP packets for LPL_PORT ──────────────────── */

/**
 * @brief Hooks NF_INET_PRE_ROUTING to take the UDP packets for LPL_PORT into the RX ring.
 *
 * @details The payload is copied from the skb at the IP header's own length (ihl * 4), which
 *          counts IP options, straight into the claimed slot. The packet then leaves the normal
 *          stack (NF_DROP), counted as dropped when the ring is full.
 */
/**
 * @brief The next RX slot, under rx_lock, or NULL when the ring is full.
 *
 * @details Reads the process's read index again only when the cached copy says the ring is full. A
 *          read index ahead of the write index, or one that frees more than the ring holds, leaves
 *          the difference out of range and the ring reads as full: whatever index the process
 *          writes, the hook writes into the slots and nowhere else.
 */
static LplRxPacket *rx_claim_slot(void)
{
    if (rx_write_index - rx_cached_read_index >= LPL_RING_SLOTS)
    {
        rx_cached_read_index = smp_load_acquire(&shm->rx.reader.read_index);
        if (rx_write_index - rx_cached_read_index >= LPL_RING_SLOTS)
            return NULL;
    }
    return &shm->rx.packets[rx_write_index & LPL_RING_MASK];
}

static unsigned int hook_ingest_packet(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct iphdr *iph;
    struct udphdr *udph;
    LplRxPacket *slot;
    uint16_t payload_len;

    (void) priv;
    (void) state;

    if (!shm)
        return NF_ACCEPT;

    iph = ip_hdr(skb);
    if (!iph || iph->protocol != IPPROTO_UDP)
        return NF_ACCEPT;

    /* Guard: reject malformed IP headers (IHL must be >= 5) */
    if (iph->ihl < 5)
        return NF_ACCEPT;

    udph = udp_hdr(skb);
    if (!udph || ntohs(udph->dest) != LPL_PORT)
        return NF_ACCEPT;

    payload_len = ntohs(udph->len) - sizeof(struct udphdr);
    if (payload_len == 0 || payload_len > LPL_MAX_PACKET_SIZE)
        return NF_ACCEPT;

    spin_lock_bh(&rx_lock);
    slot = rx_claim_slot();
    if (!slot || skb_copy_bits(skb, iph->ihl * 4u + sizeof(struct udphdr), slot->data, payload_len) < 0)
    {
        rx_drops++;
        spin_unlock_bh(&rx_lock);
        return NF_DROP;
    }

    slot->src_ip = ntohl(iph->saddr);
    slot->src_port = ntohs(udph->source);
    slot->length = payload_len;
    rx_write_index++;
    smp_store_release(&shm->rx.writer.write_index, rx_write_index);
    rx_packets++;
    rx_bytes += payload_len;
    spin_unlock_bh(&rx_lock);

    return NF_DROP; /* bypass normal stack for LPL packets */
}

/* ─── File operations ───────────────────────────────────────────────────── */

static int lpl_open(struct inode *inode, struct file *filp)
{
    (void) inode;
    (void) filp;
    return 0;
}

static int lpl_release(struct inode *inode, struct file *filp)
{
    (void) inode;
    (void) filp;
    return 0;
}

/**
 * @brief mmap handler — maps LplSharedMemory into userspace.
 *
 * Uses vmalloc_user + remap_vmalloc_range for zero-copy IPC.
 * Userspace accesses ring buffers directly without any syscalls.
 */
static int lpl_mmap(struct file *filp, struct vm_area_struct *vma)
{
    unsigned long size;
    (void) filp;

    size = vma->vm_end - vma->vm_start;

    if (size > PAGE_ALIGN(sizeof(LplSharedMemory)))
        return -EINVAL;

    if (!PAGE_ALIGNED(vma->vm_start))
        return -EINVAL;

    if (remap_vmalloc_range(vma, shm, 0) < 0)
        return -EAGAIN;

    return 0;
}

static long lpl_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    (void) filp;

    switch (cmd)
    {
    case LPL_IOCTL_RESET:
        spin_lock_bh(&rx_lock);
        rx_write_index = 0;
        rx_cached_read_index = 0;
        WRITE_ONCE(shm->rx.writer.write_index, 0U);
        WRITE_ONCE(shm->rx.reader.read_index, 0U);
        rx_packets = 0;
        rx_bytes = 0;
        rx_drops = 0;
        spin_unlock_bh(&rx_lock);
        mutex_lock(&tx_lock);
        tx_read_index = 0;
        tx_cached_write_index = 0;
        WRITE_ONCE(shm->tx.writer.write_index, 0U);
        WRITE_ONCE(shm->tx.reader.read_index, 0U);
        tx_packets = 0;
        tx_bytes = 0;
        tx_drops = 0;
        mutex_unlock(&tx_lock);
        return 0;

    case LPL_IOCTL_GET_STATS: {
        struct lpl_stats snapshot;

        spin_lock_bh(&rx_lock);
        snapshot.rx_packets = rx_packets;
        snapshot.rx_bytes = rx_bytes;
        snapshot.drops = rx_drops;
        spin_unlock_bh(&rx_lock);
        mutex_lock(&tx_lock);
        snapshot.tx_packets = tx_packets;
        snapshot.tx_bytes = tx_bytes;
        snapshot.drops += tx_drops;
        mutex_unlock(&tx_lock);
        if (copy_to_user((void __user *) arg, &snapshot, sizeof(snapshot)))
            return -EFAULT;
        return 0;
    }

    case LPL_IOCTL_KICK_TX: wake_up_interruptible(&tx_wq); return 0;

    default: return -ENOTTY;
    }
}

static const struct file_operations lpl_fops = {
    .owner = THIS_MODULE,
    .open = lpl_open,
    .release = lpl_release,
    .mmap = lpl_mmap,
    .unlocked_ioctl = lpl_ioctl,
};

/* ─── Init / Exit ───────────────────────────────────────────────────────── */

static int __init lpl_init(void)
{
    int ret;
    struct sockaddr_in bind_addr;

    /* 1. Allocate shared memory (vmalloc_user for mmap) */
    shm = vmalloc_user(sizeof(LplSharedMemory));
    if (!shm)
        return -ENOMEM;

    memset(shm, 0, sizeof(LplSharedMemory));
    shm->header.magic = LPL_MAGIC;
    shm->header.version = LPL_PROTOCOL_VERSION;
    shm->header.slots = LPL_RING_SLOTS;
    shm->header.size = sizeof(LplSharedMemory);

    /* 2. Create character device */
    ret = alloc_chrdev_region(&lpl_devno, 0, 1, LPL_DEVICE_NAME);
    if (ret < 0)
        goto fail_shm;

    cdev_init(&lpl_cdev, &lpl_fops);
    lpl_cdev.owner = THIS_MODULE;

    ret = cdev_add(&lpl_cdev, lpl_devno, 1);
    if (ret < 0)
        goto fail_region;

    lpl_class = class_create(LPL_DEVICE_NAME);
    if (IS_ERR(lpl_class))
    {
        ret = PTR_ERR(lpl_class);
        goto fail_cdev;
    }

    lpl_device = device_create(lpl_class, NULL, lpl_devno, NULL, LPL_DEVICE_NAME);
    if (IS_ERR(lpl_device))
    {
        ret = PTR_ERR(lpl_device);
        goto fail_class;
    }

    /* 3. Create kernel UDP socket for TX */
    ret = sock_create_kern(&init_net, AF_INET, SOCK_DGRAM, IPPROTO_UDP, &udp_sock);
    if (ret < 0)
    {
        pr_warn("lpl: failed to create UDP socket (%d), TX disabled\n", ret);
        udp_sock = NULL;
        /* Non-fatal: TX won't work but RX (Netfilter) still functions */
    }
    else
    {
        memset(&bind_addr, 0, sizeof(bind_addr));
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        bind_addr.sin_port = 0; /* ephemeral port */

        ret = kernel_bind(udp_sock, (void *) &bind_addr, sizeof(bind_addr));
        if (ret < 0)
            pr_warn("lpl: UDP bind failed (%d)\n", ret);
    }

    /* 4. Start TX kthread */
    init_waitqueue_head(&tx_wq);

    if (udp_sock)
    {
        tx_task = kthread_run(tx_thread_fn, NULL, "lpl_tx");
        if (IS_ERR(tx_task))
        {
            pr_warn("lpl: failed to start TX thread (%ld)\n", PTR_ERR(tx_task));
            tx_task = NULL;
        }
    }

    /* 5. Register Netfilter hook (PRE_ROUTING, capture LPL UDP packets) */
    memset(&lpl_nf_ops, 0, sizeof(lpl_nf_ops));
    lpl_nf_ops.hook = hook_ingest_packet;
    lpl_nf_ops.pf = NFPROTO_IPV4;
    lpl_nf_ops.hooknum = NF_INET_PRE_ROUTING;
    lpl_nf_ops.priority = NF_IP_PRI_FIRST;

    ret = nf_register_net_hook(&init_net, &lpl_nf_ops);
    if (ret < 0)
    {
        pr_warn("lpl: Netfilter hook registration failed (%d), RX via hook disabled\n", ret);
        /* Non-fatal: still usable via read/write fallback */
    }

    pr_info("lpl: /dev/%s registered (major %d), shm=%zu bytes\n", LPL_DEVICE_NAME, MAJOR(lpl_devno),
            sizeof(LplSharedMemory));
    return 0;

fail_class:
    class_destroy(lpl_class);
fail_cdev:
    cdev_del(&lpl_cdev);
fail_region:
    unregister_chrdev_region(lpl_devno, 1);
fail_shm:
    vfree(shm);
    shm = NULL;
    return ret;
}

static void __exit lpl_exit(void)
{
    /* Reverse order teardown */
    nf_unregister_net_hook(&init_net, &lpl_nf_ops);

    if (tx_task)
    {
        kthread_stop(tx_task);
        tx_task = NULL;
    }

    if (udp_sock)
    {
        sock_release(udp_sock);
        udp_sock = NULL;
    }

    device_destroy(lpl_class, lpl_devno);
    class_destroy(lpl_class);
    cdev_del(&lpl_cdev);
    unregister_chrdev_region(lpl_devno, 1);

    vfree(shm);
    shm = NULL;

    pr_info("lpl: /dev/%s unregistered\n", LPL_DEVICE_NAME);
}

module_init(lpl_init);
module_exit(lpl_exit);
