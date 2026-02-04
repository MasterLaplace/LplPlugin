// --- LAPLACE PLUGIN SYSTEM --- //
// File: lpl_kmod.c
// Description: Module kernel pour intercepter les paquets réseau et communiquer avec le plugin
// Auteur: MasterLaplace

#include <linux/module.h>         // Nécessaire pour tous les modules
#include <linux/kernel.h>         // Pour KERN_INFO, printk, etc.
#include <linux/netfilter.h>      // La structure des hooks Netfilter
#include <linux/netfilter_ipv4.h> // Pour les constantes IPv4 (NF_INET_PRE_ROUTING)
#include <linux/skbuff.h>         // La structure critique : struct sk_buff
#include <linux/ip.h>             // Pour struct iphdr (En-tête IP)
#include <linux/udp.h>            // Pour struct udphdr (En-tête UDP)
#include <linux/fs.h>

#include "plugin.h"

static void *k_ring_buffer = NULL;

uint32_t hook_get_engine_packet(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct iphdr *ip = ip_hdr(skb);

    if (ip->protocol != IPPROTO_UDP)
        return NF_ACCEPT;

    struct udphdr *udp = (struct udphdr *)(uint8_t *)ip + (ip->ihl * 4u);
    if (ntohs(udp->dest) == 7777u)
    {
        printk(KERN_INFO "LPL: Packet intercepted!\n");
        // Code ...
        return NF_DROP;
    }
    return NF_ACCEPT;
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
