#pragma once

#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <atomic>
#include <thread>
#include <complex>
#include <numbers>
#include <vector>
#include <cmath>
#include "lpl_protocol.h"

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
    float alphaPower = 0.0f;
    float betaPower = 0.0f;
    float concentration = 0.0f;
    bool blinkDetected = false;
};

class FastFourierTransform {
public:
    using Complex = std::complex<float>;

public:
    static void compute(std::vector<Complex> &x)
    {
        const size_t N = x.size();
        if (N <= 1u)
            return;

        uint32_t j = 0u;
        for (uint32_t i = 1u; i < N; ++i)
        {
            uint32_t bit = N >> 1u;
            for (; j & bit; bit >>= 1u)
                j ^= bit;
            j ^= bit;
            if (i < j)
                std::swap(x[i], x[j]);
        }

        for (uint32_t len = 2u; len <= N; len <<= 1u)
        {
            float ang = -2.0f * std::numbers::pi_v<float> / len;
            Complex wlen(std::cos(ang), std::sin(ang));
            for (uint32_t i = 0u; i < N; i += len)
            {
                Complex w(1.0f, 0.0f);
                for (uint32_t k = 0u; k < len / 2u; ++k)
                {
                    Complex u = x[i + k];
                    Complex v = x[i + k + len / 2u] * w;
                    x[i + k] = u + v;
                    x[i + k + len / 2u] = u - v;
                    w *= len;
                }
            }
        }
    }

    static void apply_window(std::vector<Complex> &x)
    {
        const size_t N = x.size();
        const size_t computedSize = N -1u;
        for (size_t i = 0u; i < N; ++i)
        {
            float multiplier = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> * i / computedSize));
            x[i] *= multiplier;
        }
    }
};

class OpenBCIDriver {
public:
    using Complex = std::complex<float>;

public:
    OpenBCIDriver() : _sampleIndex(0), _samplesSinceLastFFT(0), _ring(nullptr), _running(false), _fd(-1)
    {
        _ring = new BciRingBuffer();
        memset(_ring, 0, sizeof(BciRingBuffer));

        _timeDomainBuffer.resize(FFT_SIZE, 0.0f);
        _fftInput.resize(FFT_SIZE);
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
            float raw_uV = parse_channel(&pkt->data[2]);
            _timeDomainBuffer[_sampleIndex] = raw_uV;
            _sampleIndex = (_sampleIndex + 1) % FFT_SIZE;

            state.blinkDetected = (std::abs(raw_uV) > 150.0f);
            _samplesSinceLastFFT++;

            if (_samplesSinceLastFFT >= UPDATE_INTERVAL)
            {
                processFFT(state);
                _samplesSinceLastFFT = 0u;
            }

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

            if (buffer[0] == 0xA0 && buffer[BCI_PACKET_SIZE - 1u] == 0xC0)
            {
                uint32_t tail = smp_load_acquire(&_ring->idx.tail);
                uint32_t head = _ring->idx.head;
                uint32_t next_head = (head + 1u) & (BCI_RING_SLOTS - 1u);

                if (next_head != tail)
                {
                    BciPacket *pkt = &_ring->packets[head];
                    memcpy(pkt->data, buffer, BCI_PACKET_SIZE);
                    smp_store_release(&_ring->idx.head, next_head);
                }
                continue;
            }

            uint8_t byte = 0u;
            while (_running && read(_fd, &byte, 1u) > 0 && byte == 0xA0);

            buffer[0] = 0xA0;

            uint8_t needed = BCI_PACKET_SIZE - 1u;
            uint8_t received = 0u;
            while (received < needed && _running)
            {
                int r = read(_fd, buffer + 1u + received, needed - received);
                if (r > 0)
                    received += r;
            }
        }
    }

    void processFFT(NeuralState &state)
    {
        for (size_t i = 0u; i < FFT_SIZE; ++i)
        {
            size_t idx = (_sampleIndex + i) % FFT_SIZE;
            _fftInput[i] = Complex(_timeDomainBuffer[idx], 0.0f);
        }

        FastFourierTransform::apply_window(_fftInput);
        FastFourierTransform::compute(_fftInput);

        float alphaSum = 0.0f;
        float betaSum = 0.0f;
        const float normFactor = 2.0f / FFT_SIZE;

        for (size_t i = 1u; i < FFT_SIZE / 2u; ++i)
        {
            float freq = i * FREQ_RES;
            float magnitude = std::abs(_fftInput[i]) * normFactor;

            if (freq >= 8.0f && freq <= 12.0f)
                alphaSum += magnitude;
            else if (freq >= 13.0f && freq <= 30.0f)
                betaSum += magnitude;
        }

        float smoothFactor = 0.1f;
        state.alphaPower = state.alphaPower * (1.0f - smoothFactor) + alphaSum * smoothFactor;
        state.betaPower  = state.betaPower  * (1.0f - smoothFactor) + betaSum  * smoothFactor;

        float totalPower = state.alphaPower + state.betaPower + 0.0001f;
        float ratio = state.betaPower / totalPower;

        state.concentration = state.concentration * 0.9f + ratio * 0.1f;
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

    float parse_channel(const uint8_t *data)
    {
        int32_t value = (data[0] << 16) | (data[1] << 8) | data[2];
        if (value & 0x00800000)
            value |= 0xFF000000;
        return (float)value * BCI_SCALE_FACTOR;
    }

private:
    static constexpr float BCI_SCALE_FACTOR = 4.5f / 24.0f / 8388607.0f * 1000000.0f;
    static constexpr size_t UPDATE_INTERVAL = 32u;
    static constexpr size_t FFT_SIZE = 256u;
    static constexpr float SAMPLE_RATE = 250.0f;
    static constexpr float FREQ_RES = SAMPLE_RATE / FFT_SIZE;
    std::vector<float> _timeDomainBuffer;
    std::vector<Complex> _fftInput;
    size_t _sampleIndex;
    size_t _samplesSinceLastFFT;
    BciRingBuffer *_ring;
    std::atomic<bool> _running;
    std::thread _worker;
    int _fd;
};
