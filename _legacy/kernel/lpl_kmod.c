// --- LAPLACE PLUGIN SYSTEM --- //
// File: lpl_kmod.c
// Description: Module kernel RX/TX UDP via Shared Memory RingBuffer
// Auteur: MasterLaplace

#include <linux/init.h>
#include <linux/module.h>         // MODULE_LICENSE, MODULE_AUTHOR, MODULE_DESCRIPTION, module_init, module_exit
#include <linux/kernel.h>         // KERN_INFO, printk, pr_info, pr_err
#include <linux/netfilter.h>      // struct nf_hook_ops, NF_INET_PRE_ROUTING, NF_IP_PRI_FIRST
#include <linux/netfilter_ipv4.h> // NF_INET_PRE_ROUTING, nf_register_net_hook, nf_unregister_net_hook, struct nf_hook_state
#include <linux/skbuff.h>         // struct sk_buff, skb_copy_bits
#include <linux/ip.h>             // struct iphdr, ip_hdr
#include <linux/udp.h>            // struct udphdr, ntohs
#include <linux/fs.h>             // struct file_operations, register_chrdev, unregister_chrdev
#include <linux/device.h>         //  class_create, device_create
#include <linux/err.h>            //  IS_ERR, PTR_ERR
#include <linux/socket.h>
#include <linux/in.h>

#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mm.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/sched.h>
#include <net/sock.h>

#include "lpl_protocol.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("MasterLaplace");
MODULE_DESCRIPTION("Laplace Kernel Module: High-Perf Zero-Copy RingBuffer RX/TX");

static LplSharedMemory *shm = NULL;
static int driver_id = 0;
static struct class *lpl_class = NULL;
static struct device *instance = NULL;

static struct task_struct *tx_task = NULL;
static wait_queue_head_t tx_wq;
static struct socket *udp_sock = NULL;

static int send_udp_packet(TxPacket *pkt)
{
    struct msghdr msg;
    struct kvec iov;
    struct sockaddr_in to;
    int len = pkt->length;

    if (!udp_sock)
        return -EIO;

    memset(&to, 0, sizeof(to));
    to.sin_family = AF_INET;
    to.sin_addr.s_addr = pkt->dst_ip;
    to.sin_port = pkt->dst_port;

    memset(&msg, 0, sizeof(msg));
    msg.msg_name = &to;
    msg.msg_namelen = sizeof(to);

    iov.iov_base = pkt->data;
    iov.iov_len = len;

    return kernel_sendmsg(udp_sock, &msg, &iov, 1, len);
}

static int tx_thread_fn(void *data)
{
    TxRingBuffer *ring = &shm->tx;
    uint32_t head, tail;

    while (!kthread_should_stop())
    {
        tail = smp_load_acquire(&ring->idx.tail);
        head = ring->idx.head;

        wait_event_interruptible(tx_wq, (head != smp_load_acquire(&ring->idx.tail)) || kthread_should_stop());

        if (kthread_should_stop())
            break;

        tail = smp_load_acquire(&ring->idx.tail);

        while (head != tail)
        {
            TxPacket *pkt = &ring->packets[head];
            send_udp_packet(pkt);
            head = (head + 1u) & (RING_SLOTS - 1u);
            smp_store_release(&ring->idx.head, head);
        }
        cond_resched();
    }
    return 0;
}

static uint32_t hook_ingest_packet(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct iphdr *ip = ip_hdr(skb);

    if (ip->protocol != IPPROTO_UDP)
        return NF_ACCEPT;

    uint32_t ip_len = ip->ihl * 4u;
    struct udphdr *udp = (struct udphdr *)((uint8_t *)ip + ip_len);
    if (!shm || ntohs(udp->dest) != LPL_PORT)
        return NF_ACCEPT;

    RxRingBuffer *ring = &shm->rx;
    uint32_t payload_len = ntohs(udp->len) - sizeof(struct udphdr);

    if (payload_len > MAX_PACKET_SIZE)
        return NF_DROP;

    uint32_t tail = smp_load_acquire(&ring->idx.tail);
    uint32_t head = ring->idx.head;
    uint32_t next_head = (head + 1u) & (RING_SLOTS - 1u);

    if (next_head == tail)
        return NF_DROP;

    RxPacket *pkt = &ring->packets[head];
    pkt->src_ip = ip->saddr;
    pkt->src_port = udp->source;
    pkt->length = payload_len;

    if (skb_copy_bits(skb, ip_len + sizeof(struct udphdr), pkt->data, payload_len) < 0)
        return NF_DROP;

    smp_store_release(&ring->idx.head, next_head);
    return NF_DROP;
}

static int lpl_open(struct inode *inode, struct file *file)
{
    return 0;
}

static int lpl_release(struct inode *inode, struct file *file)
{
    return 0;
}

static int lpl_mmap(struct file *file, struct vm_area_struct *vma)
{
    unsigned long length = vma->vm_end - vma->vm_start;
    unsigned long process_size = PAGE_ALIGN(sizeof(LplSharedMemory));

    if (length > process_size)
        return -EINVAL;

    return remap_vmalloc_range(vma, shm, 0);
}

static long lpl_ioctl(struct file *file, uint32_t cmd, unsigned long arg)
{
    if (cmd != LPL_IOC_KICK_TX)
        return -EINVAL;
    wake_up_interruptible(&tx_wq);
    return 0;
}

static const struct file_operations lpl_fops = {
    .owner = THIS_MODULE,
    .open = lpl_open,
    .release = lpl_release,
    .mmap = lpl_mmap,
    .unlocked_ioctl = lpl_ioctl
};

static const struct nf_hook_ops lpl_nf_ops = {
    .hook = hook_ingest_packet,
    .pf = NFPROTO_IPV4,
    .hooknum = NF_INET_PRE_ROUTING,
    .priority = NF_IP_PRI_FIRST
};

static int __init lpl_init(void)
{
    pr_info("[LPL] ========== Module Init ==========\n");
    int result = 0;

    shm = (LplSharedMemory *)vmalloc_user(sizeof(LplSharedMemory));
    if (!shm)
        return -ENOMEM;

    memset(shm, 0, sizeof(LplSharedMemory));

    init_waitqueue_head(&tx_wq);

    if ((result = sock_create_kern(&init_net, PF_INET, SOCK_DGRAM, IPPROTO_UDP, &udp_sock)) < 0)
    {
        pr_err("[LPL] Fail to create UDP kernel socket\n");
        goto error_memory;
    }

    if ((driver_id = register_chrdev(0, LPL_DEVICE_NAME, &lpl_fops)) < 0)
    {
        pr_err("[LPL] Failed to register char device\n");
        result = driver_id;
        goto error_socket;
    }

    if (IS_ERR(lpl_class = class_create(LPL_CLASS_NAME)))
    {
        pr_err("[LPL] Failed to create class\n");
        result = PTR_ERR(lpl_class);
        goto error_chrdev;
    }

    if (IS_ERR(instance = device_create(lpl_class, instance, MKDEV(driver_id, 0), NULL, LPL_DEVICE_NAME)))
    {
        pr_err("[LPL] Failed to create device instance\n");
        result = PTR_ERR(instance);
        goto error_class;
    }

    if ((result = nf_register_net_hook(&init_net, &lpl_nf_ops)) < 0)
    {
        pr_err("[LPL] Failed to register Netfilter hook\n");
        goto error_device;
    }

    if (IS_ERR(tx_task = kthread_run(tx_thread_fn, NULL, "lpl_tx_worker")))
    {
        pr_err("[LPL] Failed to run kernel thread\n");
        result = PTR_ERR(tx_task);
        goto error_nf;
    }

    pr_info("[LPL] Netfilter hook registered (NF_INET_PRE_ROUTING, port %d)\n", LPL_PORT);
    pr_info("[LPL] ===== Module Loaded Successfully =====\n");
    return 0;

error_nf: nf_unregister_net_hook(&init_net, &lpl_nf_ops);
error_device: device_destroy(lpl_class, MKDEV(driver_id, 0));
error_class: class_destroy(lpl_class);
error_chrdev: unregister_chrdev(driver_id, LPL_DEVICE_NAME);
error_socket: sock_release(udp_sock);
error_memory: vfree(shm);
    return result;
}

static void __exit lpl_exit(void)
{
    pr_info("[LPL] ========== Module Exit ==========\n");

    if (tx_task)
        kthread_stop(tx_task);

    nf_unregister_net_hook(&init_net, &lpl_nf_ops);
    device_destroy(lpl_class, MKDEV(driver_id, 0));
    class_destroy(lpl_class);
    unregister_chrdev(driver_id, LPL_DEVICE_NAME);

    if (udp_sock)
        sock_release(udp_sock);

    if (shm)
    {
        vfree(shm);
        shm = NULL;
    }
    pr_info("[LPL] ===== Module Unloaded =====\n");
}

module_init(lpl_init);
module_exit(lpl_exit);
