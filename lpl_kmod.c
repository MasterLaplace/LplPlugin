// --- LAPLACE PLUGIN SYSTEM --- //
// File: lpl_kmod.c
// Description: Module kernel pour intercepter les paquets réseau et communiquer avec le plugin
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

#include "lpl_protocol.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("MasterLaplace");
MODULE_DESCRIPTION("Laplace Kernel Module for Zero-Copy Network Ingestion");

static void *k_ring_buffer = NULL;
static int driver_id = 0;
static struct class *lpl_class = NULL;
static struct device *instance = NULL;
#ifdef LPL_MONITORING
static uint32_t packet_count_intercepted = 0;
#endif

static uint32_t hook_get_engine_packet(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct iphdr *ip = ip_hdr(skb);

    if (ip->protocol != IPPROTO_UDP)
        return NF_ACCEPT;

    uint32_t ip_len = ip->ihl * 4u;
    struct udphdr *udp = (struct udphdr *)((uint8_t *)ip + ip_len);
    if (ntohs(udp->dest) != 7777u || !k_ring_buffer)
        return NF_ACCEPT;

    NetworkRingBuffer *ring = (NetworkRingBuffer *)k_ring_buffer;
    uint32_t payload_len = ntohs(udp->len) - sizeof(struct udphdr);

    if (payload_len > MAX_PACKET_SIZE)
        return printk(KERN_WARNING "[LPL] Packet too big (%u > %u)\n", payload_len, MAX_PACKET_SIZE), NF_DROP;

    uint32_t head = atomic_read(&ring->head);
    uint32_t tail = atomic_read(&ring->tail);
    uint32_t next_head = (head + 1u) & (RING_SIZE - 1u);

    if (next_head == tail)
        return printk(KERN_WARNING "[LPL] Ring buffer FULL (head=%u, tail=%u)\n", head, tail), NF_DROP;

    DynamicPacket *pkt = &ring->packets[head];

    if (skb_copy_bits(skb, ip_len + sizeof(struct udphdr), pkt->data, payload_len) < 0)
        return printk(KERN_WARNING "[LPL] skb_copy_bits failed\n"), NF_DROP;

    pkt->msgType = RING_MSG_DYNAMIC;
    pkt->size = (uint16_t)payload_len;
    atomic_set(&ring->head, next_head);

#ifdef LPL_MONITORING
    packet_count_intercepted++;
    if (packet_count_intercepted % 10 == 0)
    {
        printk(KERN_INFO "[LPL] Intercepted %u packets | head: %u → %u | payload: %u bytes\n",
               packet_count_intercepted, head, next_head, payload_len);
    }
#endif

    return NF_DROP;
}

static int lpl_open(struct inode *inode, struct file *file)
{
    if (k_ring_buffer)
        return -EEXIST;
    k_ring_buffer = kzalloc(sizeof(NetworkRingBuffer), GFP_KERNEL);
    if (!k_ring_buffer)
        return -ENOMEM;
    return 0;
}

static int lpl_release(struct inode *inode, struct file *file)
{
    if (!k_ring_buffer)
        return -EFAULT;
    kfree(k_ring_buffer);
    k_ring_buffer = NULL;
    return 0;
}

static int lpl_mmap(struct file *file, struct vm_area_struct *vma)
{
    if (!k_ring_buffer)
        return -EFAULT;
    if (remap_pfn_range(vma, vma->vm_start, virt_to_phys(k_ring_buffer) >> PAGE_SHIFT,
                        sizeof(NetworkRingBuffer), vma->vm_page_prot))
        return -EAGAIN;
    return 0;
}

static const struct file_operations lpl_fops = {
    .owner = THIS_MODULE,
    .open = lpl_open,
    .release = lpl_release,
    .mmap = lpl_mmap
};

static const struct nf_hook_ops lpl_nf_ops = {
    .hook = hook_get_engine_packet,
    .pf = NFPROTO_IPV4,
    .hooknum = NF_INET_PRE_ROUTING,
    .priority = NF_IP_PRI_FIRST
};

static int __init lpl_init(void)
{
    pr_info("[LPL] ========== Module Init ==========\n");
    int result = 0;

    if ((driver_id = register_chrdev(0, "lpl_driver", &lpl_fops)) < 0)
    {
        pr_err("[LPL] Failed to register char device\n");
        return driver_id;
    }

    if (IS_ERR(lpl_class = class_create("lpl")))
    {
        pr_err("[LPL] Failed to create class\n");
        result = PTR_ERR(lpl_class);
        goto error_chrdev;
    }

    if (IS_ERR(instance = device_create(lpl_class, instance, MKDEV(driver_id, 0), NULL, "lpl_driver")))
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

    pr_info("[LPL] Netfilter hook registered (NF_INET_PRE_ROUTING, port 7777)\n");
    pr_info("[LPL] ===== Module Loaded Successfully =====\n");
    return 0;

error_device: device_destroy(lpl_class, MKDEV(driver_id, 0));
error_class: class_destroy(lpl_class);
error_chrdev: unregister_chrdev(driver_id, "lpl_driver");
    return result;
}

static void __exit lpl_exit(void)
{
    pr_info("[LPL] ========== Module Exit ==========\n");

    nf_unregister_net_hook(&init_net, &lpl_nf_ops);
    device_destroy(lpl_class, MKDEV(driver_id, 0));
    class_destroy(lpl_class);
    unregister_chrdev(driver_id, "lpl_driver");

    if (k_ring_buffer)
    {
        kfree(k_ring_buffer);
        k_ring_buffer = NULL;
    }

#ifdef LPL_MONITORING
    pr_info("[LPL] Total packets intercepted: %u\n", packet_count_intercepted);
#endif
    pr_info("[LPL] ===== Module Unloaded =====\n");
}

module_init(lpl_init);
module_exit(lpl_exit);
