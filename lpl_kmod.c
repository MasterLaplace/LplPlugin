#include <linux/module.h>       // Nécessaire pour tous les modules
#include <linux/kernel.h>       // Pour KERN_INFO, printk, etc.
#include <linux/netfilter.h>    // La structure des hooks Netfilter
#include <linux/netfilter_ipv4.h> // Pour les constantes IPv4 (NF_INET_PRE_ROUTING)
#include <linux/skbuff.h>       // La structure critique : struct sk_buff
#include <linux/ip.h>           // Pour struct iphdr (En-tête IP)
#include <linux/udp.h>          // Pour struct udphdr (En-tête UDP)
#include <stdint.h>

uint32_t hook_get_engine_packet(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct iphdr *ip = ip_hdr(skb);

    if (ip->protocol != IPPROTO_UDP)
        return NF_ACCEPT;

    struct udphdr *udp = (struct udphdr *)(uint8_t*)ip + (ip->ihl * 4);
    if (ntohs(udp->dest) == 7777u)
    {
        printk(KERN_INFO "LPL: Packet intercepted!\n");
        // Code ...
        return NF_DROP;
    }
    return NF_ACCEPT;
}
