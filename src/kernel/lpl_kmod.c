// /////////////////////////////////////////////////////////////////////////////
/// @file lpl_kmod.c
/// @brief Linux kernel module — char device for ultra-low-latency IPC (C17).
///
/// Exposes /dev/lpl0 as a character device. Userspace writes packets in;
/// the kernel rings them through a lock-free SPSC ring buffer that can
/// be mmap'd for zero-copy reads.
///
/// Build: part of the out-of-tree module build (Kbuild / Makefile).
// /////////////////////////////////////////////////////////////////////////////

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>

#include "lpl_protocol.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Laplace");
MODULE_DESCRIPTION("LplPlugin ultra-low-latency IPC char device");
MODULE_VERSION("0.1");

// ─────────────────────────────────────────────────────────────────────────────
// Module-level state
// ─────────────────────────────────────────────────────────────────────────────

static dev_t           lpl_devno;
static struct cdev     lpl_cdev;
static struct class   *lpl_class;
static struct device  *lpl_device;

static DEFINE_MUTEX(lpl_lock);

static struct lpl_stats stats;

static struct lpl_ring_slot *ring;
static uint32_t ring_head;   /* writer (userspace write → kernel) */
static uint32_t ring_tail;   /* reader (kernel → userspace read)  */

// ─────────────────────────────────────────────────────────────────────────────
// File operations
// ─────────────────────────────────────────────────────────────────────────────

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

static ssize_t lpl_read(struct file *filp, char __user *buf,
                         size_t count, loff_t *off)
{
    struct lpl_ring_slot *slot;
    uint32_t tail;
    (void)filp;
    (void)off;

    mutex_lock(&lpl_lock);

    tail = ring_tail;
    if (tail == ring_head)
    {
        mutex_unlock(&lpl_lock);
        return 0; /* ring empty */
    }

    slot = &ring[tail % LPL_RING_SLOTS];

    if (count < slot->length)
    {
        mutex_unlock(&lpl_lock);
        return -EINVAL;
    }

    if (copy_to_user(buf, slot->data, slot->length))
    {
        mutex_unlock(&lpl_lock);
        return -EFAULT;
    }

    stats.rx_packets++;
    stats.rx_bytes += slot->length;

    ring_tail = tail + 1;

    mutex_unlock(&lpl_lock);

    return (ssize_t)slot->length;
}

static ssize_t lpl_write(struct file *filp, const char __user *buf,
                          size_t count, loff_t *off)
{
    struct lpl_ring_slot *slot;
    uint32_t head;
    (void)filp;
    (void)off;

    if (count == 0 || count > LPL_MAX_PACKET_SIZE)
    {
        return -EINVAL;
    }

    mutex_lock(&lpl_lock);

    head = ring_head;
    if ((head - ring_tail) >= LPL_RING_SLOTS)
    {
        stats.drops++;
        mutex_unlock(&lpl_lock);
        return -ENOSPC; /* ring full */
    }

    slot = &ring[head % LPL_RING_SLOTS];
    slot->length = (uint32_t)count;

    if (copy_from_user(slot->data, buf, count))
    {
        mutex_unlock(&lpl_lock);
        return -EFAULT;
    }

    stats.tx_packets++;
    stats.tx_bytes += count;

    ring_head = head + 1;

    mutex_unlock(&lpl_lock);

    return (ssize_t)count;
}

static long lpl_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    (void)filp;

    switch (cmd)
    {
    case LPL_IOCTL_RESET:
        mutex_lock(&lpl_lock);
        ring_head = 0;
        ring_tail = 0;
        memset(&stats, 0, sizeof(stats));
        mutex_unlock(&lpl_lock);
        return 0;

    case LPL_IOCTL_GET_STATS:
        if (copy_to_user((void __user *)arg, &stats, sizeof(stats)))
        {
            return -EFAULT;
        }
        return 0;

    default:
        return -ENOTTY;
    }
}

static const struct file_operations lpl_fops = {
    .owner          = THIS_MODULE,
    .open           = lpl_open,
    .release        = lpl_release,
    .read           = lpl_read,
    .write          = lpl_write,
    .unlocked_ioctl = lpl_ioctl,
};

// ─────────────────────────────────────────────────────────────────────────────
// Init / Exit
// ─────────────────────────────────────────────────────────────────────────────

static int __init lpl_init(void)
{
    int ret;

    ring = kzalloc(sizeof(*ring) * LPL_RING_SLOTS, GFP_KERNEL);
    if (!ring)
    {
        return -ENOMEM;
    }

    ret = alloc_chrdev_region(&lpl_devno, 0, 1, LPL_DEVICE_NAME);
    if (ret < 0)
    {
        goto fail_ring;
    }

    cdev_init(&lpl_cdev, &lpl_fops);
    lpl_cdev.owner = THIS_MODULE;

    ret = cdev_add(&lpl_cdev, lpl_devno, 1);
    if (ret < 0)
    {
        goto fail_region;
    }

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

    pr_info("lpl: /dev/%s registered (major %d)\n",
            LPL_DEVICE_NAME, MAJOR(lpl_devno));
    return 0;

fail_class:
    class_destroy(lpl_class);
fail_cdev:
    cdev_del(&lpl_cdev);
fail_region:
    unregister_chrdev_region(lpl_devno, 1);
fail_ring:
    kfree(ring);
    return ret;
}

static void __exit lpl_exit(void)
{
    device_destroy(lpl_class, lpl_devno);
    class_destroy(lpl_class);
    cdev_del(&lpl_cdev);
    unregister_chrdev_region(lpl_devno, 1);
    kfree(ring);
    pr_info("lpl: /dev/%s unregistered\n", LPL_DEVICE_NAME);
}

module_init(lpl_init);
module_exit(lpl_exit);
