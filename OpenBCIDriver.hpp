#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <atomic>
#include <thread>
#include "lpl_protocol.h"

#pragma once

#define BCI_PACKET_SIZE 33
#define BCI_RING_SLOTS 1024

struct BciPacket {
    uint8_t data[BCI_PACKET_SIZE];
};

struct BciRingBuffer{
    RingHeader idx;
    BciPacket packets[BCI_RING_SLOTS];
};

struct NeuralState {
    float concentration = 0.0f;
    bool blinkDetected = false;
};

class OpenBCIDriver {
public:
    OpenBCIDriver() : _ring(nullptr), _running(false), _fd(-1)
    {
        _ring = new BciRingBuffer();
        memset(_ring, 0, sizeof(BciRingBuffer));
    }

    ~OpenBCIDriver()
    {
        stop();
        delete _ring;
    }

    bool init(const char *port = "/dev/ttyUSB0")
    {
        if ((_fd = setup_serial_port(port)) < 0)
            return false;
        start();
        return true;
    }

    void start()
    {
        _running = true;
        _worker = std::thread(&OpenBCIDriver::worker_loop, this);
    }

    void stop()
    {
        _running = false;
        if (_worker.joinable())
            _worker.join();
        if (_fd >= 0)
        {
            close(_fd);
            _fd = -1;
        }
    }

    void update(NeuralState &state)
    {
        uint32_t head = smp_load_acquire(&_ring->idx.head);
        uint32_t tail = _ring->idx.tail;

        while (head != tail)
        {
            BciPacket *pkt = &_ring->packets[tail];
            uint8_t rawVal = pkt->data[2];
            state.concentration = (float)rawVal / 255.0f;
            tail = (tail + 1u) & (BCI_RING_SLOTS - 1u);
        }
        smp_store_release(&_ring->idx.tail, tail);
    }

private:
    void worker_loop()
    {
        uint8_t buffer[BCI_PACKET_SIZE];

        while (_running)
        {
            int n = read(_fd, buffer, BCI_PACKET_SIZE);
            if (n != BCI_PACKET_SIZE)
                continue;

            uint32_t tail = smp_load_acquire(&_ring->idx.tail);
            uint32_t head = _ring->idx.head;
            uint32_t next_head = (head + 1u) & (BCI_RING_SLOTS - 1u);

            if (next_head == tail)
                continue;

            BciPacket *pkt = &_ring->packets[head];
            memcpy(pkt->data, buffer, BCI_PACKET_SIZE);
            smp_store_release(&_ring->idx.head, next_head);
        }
    }

    int setup_serial_port(const char *device_path)
    {
        int fd = open(device_path, O_RDWR | O_NOCTTY);
        if (fd == -1)
            return printf("[SERIAL] Failed to open port: %s\n", strerror(errno)), -1;

        if (ioctl(fd, TIOCEXCL, NULL) < 0)
        {
            printf("[SERIAL] Failed to set exclusive access\n");
            close(fd);
            return -1;
        }

        struct termios tty;
        if (tcgetattr(fd, &tty) != 0)
        {
            printf("[SERIAL] Error from tcgetattr: %s\n", strerror(errno));
            close(fd);
            return -1;
        }

        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cflag |= CREAD | CLOCAL;
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;
        tty.c_oflag &= ~ONLCR;

        tty.c_cc[VMIN] = 33u;
        tty.c_cc[VTIME] = 1u;

        cfsetispeed(&tty, B115200);
        cfsetospeed(&tty, B115200);

        if (tcsetattr(fd, TCSANOW, &tty) != 0)
        {
            printf("[SERIAL] Error from tcsetattr: %s\n", strerror(errno));
            close(fd);
            return -1;
        }

        return fd;
    }

private:
    BciRingBuffer *_ring;
    std::atomic<bool> _running;
    std::thread _worker;
    int _fd;
};
