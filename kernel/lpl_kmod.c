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

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/net.h>
#include <linux/in.h>
#include <linux/inet.h>
#include <linux/socket.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/skbuff.h>

#include "lpl_protocol.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("MasterLaplace");
MODULE_DESCRIPTION("Laplace Kernel Module: High-Perf Zero-Copy RingBuffer RX/TX");
MODULE_VERSION("0.2");

/* ─── Module state ──────────────────────────────────────────────────────── */

static dev_t               lpl_devno;
static struct cdev         lpl_cdev;
static struct class       *lpl_class;
static struct device      *lpl_device;

static LplSharedMemory    *shm;            /* vmalloc_user shared memory    */
static struct lpl_stats    stats;

static struct task_struct *tx_task;         /* TX kthread                    */
static wait_queue_head_t   tx_wq;          /* wait queue for TX kick        */

static struct socket      *udp_sock;       /* kernel-space UDP socket       */

static struct nf_hook_ops  lpl_nf_ops;     /* Netfilter hook registration   */

/* ─── UDP send from kernel space ────────────────────────────────────────── */

static int send_udp_packet(LplTxPacket *pkt)
{
    struct msghdr msg = {};
    struct kvec   iov;
    struct sockaddr_in dst;

    if (!udp_sock || pkt->length == 0 || pkt->length > LPL_MAX_PACKET_SIZE)
        return -EINVAL;

    memset(&dst, 0, sizeof(dst));
    dst.sin_family      = AF_INET;
    dst.sin_port        = htons(pkt->dst_port);
    dst.sin_addr.s_addr = htonl(pkt->dst_ip);

    iov.iov_base = pkt->data;
    iov.iov_len  = pkt->length;

    msg.msg_name    = &dst;
    msg.msg_namelen = sizeof(dst);

    return kernel_sendmsg(udp_sock, &msg, &iov, 1, pkt->length);
}

/* ─── TX kthread ────────────────────────────────────────────────────────── */

/**
 * TX thread drains the TX ring and sends packets via kernel UDP socket.
 * Sleeps via wait_event_interruptible until kicked by ioctl or signaled.
 */
static int tx_thread_fn(void *data)
{
    (void)data;

    while (!kthread_should_stop())
    {
        uint32_t head, tail;
        uint32_t loop_safety = LPL_RING_SLOTS * 2;

        wait_event_interruptible(tx_wq,
            kthread_should_stop() ||
            (smp_load_acquire(&shm->tx.idx.head) !=
             smp_load_acquire(&shm->tx.idx.tail)));

        if (kthread_should_stop())
            break;

        head = smp_load_acquire(&shm->tx.idx.head);
        tail = smp_load_acquire(&shm->tx.idx.tail);

        while (tail != head && loop_safety-- > 0)
        {
            LplTxPacket *pkt = &shm->tx.packets[tail & LPL_RING_MASK];
            int ret = send_udp_packet(pkt);

            if (ret >= 0)
            {
                stats.tx_packets++;
                stats.tx_bytes += pkt->length;
            }
            else
            {
                stats.drops++;
            }

            tail++;
            smp_store_release(&shm->tx.idx.tail, tail);
            cond_resched();
        }
    }

    return 0;
}

/* ─── Netfilter hook: capture UDP packets for LPL_PORT ──────────────────── */

/**
 * Hooks NF_INET_PRE_ROUTING to intercept UDP packets destined for LPL_PORT.
 * Copies the payload directly into the RX ring via skb_copy_bits (zero-copy
 * from skb). Returns NF_DROP to bypass the normal network stack.
 */
static unsigned int hook_ingest_packet(void *priv,
                                       struct sk_buff *skb,
                                       const struct nf_hook_state *state)
{
    struct iphdr  *iph;
    struct udphdr *udph;
    uint32_t       head, tail, next;
    LplRxPacket   *slot;
    uint16_t       payload_len;

    (void)priv;
    (void)state;

    if (!shm)
        return NF_ACCEPT;

    iph = ip_hdr(skb);
    if (!iph || iph->protocol != IPPROTO_UDP)
        return NF_ACCEPT;

    udph = udp_hdr(skb);
    if (!udph || ntohs(udph->dest) != LPL_PORT)
        return NF_ACCEPT;

    payload_len = ntohs(udph->len) - sizeof(struct udphdr);
    if (payload_len == 0 || payload_len > LPL_MAX_PACKET_SIZE)
        return NF_ACCEPT;

    /* Lockless SPSC: producer = Netfilter (this hook), consumer = userspace */
    head = smp_load_acquire(&shm->rx.idx.head);
    tail = smp_load_acquire(&shm->rx.idx.tail);
    next = head + 1;

    /* Ring full check */
    if ((next - tail) > LPL_RING_SLOTS)
    {
        stats.drops++;
        return NF_DROP;
    }

    slot = &shm->rx.packets[head & LPL_RING_MASK];
    slot->src_ip   = ntohl(iph->saddr);
    slot->src_port = ntohs(udph->source);
    slot->length   = payload_len;

    /* Copy payload from SKB directly into ring slot */
    if (skb_copy_bits(skb, sizeof(struct iphdr) + sizeof(struct udphdr),
                      slot->data, payload_len) < 0)
    {
        stats.drops++;
        return NF_DROP;
    }

    smp_store_release(&shm->rx.idx.head, next);
    stats.rx_packets++;
    stats.rx_bytes += payload_len;

    return NF_DROP; /* bypass normal stack for LPL packets */
}

/* ─── File operations ───────────────────────────────────────────────────── */

static int lpl_open(struct inode *inode, struct file *filp)
{
    (void)inode;
    (void)filp;
    return 0;
}

static int lpl_release(struct inode *inode, struct file *filp)
{
    (void)inode;
    (void)filp;
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
    (void)filp;

    size = vma->vm_end - vma->vm_start;

    if (size > sizeof(LplSharedMemory))
        return -EINVAL;

    if (!PAGE_ALIGNED(vma->vm_start))
        return -EINVAL;

    if (remap_vmalloc_range(vma, shm, 0) < 0)
        return -EAGAIN;

    return 0;
}

static long lpl_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    (void)filp;

    switch (cmd)
    {
    case LPL_IOCTL_RESET:
        if (shm)
        {
            shm->rx.idx.head = 0;
            shm->rx.idx.tail = 0;
            shm->tx.idx.head = 0;
            shm->tx.idx.tail = 0;
        }
        memset(&stats, 0, sizeof(stats));
        return 0;

    case LPL_IOCTL_GET_STATS:
        if (copy_to_user((void __user *)arg, &stats, sizeof(stats)))
            return -EFAULT;
        return 0;

    case LPL_IOCTL_KICK_TX:
        wake_up_interruptible(&tx_wq);
        return 0;

    default:
        return -ENOTTY;
    }
}

static const struct file_operations lpl_fops = {
    .owner          = THIS_MODULE,
    .open           = lpl_open,
    .release        = lpl_release,
    .mmap           = lpl_mmap,
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
        bind_addr.sin_family      = AF_INET;
        bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        bind_addr.sin_port        = 0; /* ephemeral port */

        ret = kernel_bind(udp_sock, (struct sockaddr *)&bind_addr, sizeof(bind_addr));
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
    lpl_nf_ops.hook     = hook_ingest_packet;
    lpl_nf_ops.pf       = NFPROTO_IPV4;
    lpl_nf_ops.hooknum  = NF_INET_PRE_ROUTING;
    lpl_nf_ops.priority = NF_IP_PRI_FIRST;

    ret = nf_register_net_hook(&init_net, &lpl_nf_ops);
    if (ret < 0)
    {
        pr_warn("lpl: Netfilter hook registration failed (%d), RX via hook disabled\n", ret);
        /* Non-fatal: still usable via read/write fallback */
    }

    pr_info("lpl: /dev/%s registered (major %d), shm=%zu bytes\n",
            LPL_DEVICE_NAME, MAJOR(lpl_devno), sizeof(LplSharedMemory));
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
