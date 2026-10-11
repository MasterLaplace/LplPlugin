/**
 * @file RingBench.cpp
 * @brief The ring section of lpl-benchmark.
 * @see   RingBench.hpp
 */

#include <lpl/bench/RingBench.hpp>

#include <lpl/bench/Harness.hpp>
#include <lpl/container/RingBuffer.hpp>
#include <lpl/core/Platform.hpp>
#include <lpl/core/Types.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#    include <fstream>
#    include <pthread.h>
#    include <sched.h>
#    include <string>
#endif

namespace lpl::bench {

namespace {

constexpr core::u32 kElementCount = 1u << 16u;
constexpr core::usize kSlots = 1024u;
constexpr core::u32 kBatch = 64u;
constexpr core::u32 kRoundTrips = 1u << 12u;
constexpr core::u32 kWarmupPairs = 5u;
constexpr core::u32 kPairs = 101u;

/** @brief A message of one cache line, the size of a packet header or a small event. */
struct Message {
    core::u32 sequence = 0u; /**< Position in the stream. */
    core::u32 body[15] = {}; /**< Payload the consumer does not read. */
};

static_assert(sizeof(Message) == 64u, "a message fills one cache line");

/**
 * @brief The ring as it was before LplKernel#272, kept as the baseline every ring change is measured
 *        against: wrapped indices with one slot left empty, both indices on one line, and the other
 *        end's index read with acquire on every push and every pop.
 */
template <typename T, core::usize Capacity> class SeedRing final {
public:
    [[nodiscard]] bool push(const T &item)
    {
        const core::usize tail = _tail.load(std::memory_order_relaxed);
        const core::usize next = (tail + 1u) & kMask;

        if (next == _head.load(std::memory_order_acquire))
            return false;
        _buffer[tail] = item;
        _tail.store(next, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool pop(T &item)
    {
        const core::usize head = _head.load(std::memory_order_relaxed);

        if (head == _tail.load(std::memory_order_acquire))
            return false;
        item = _buffer[head];
        _head.store((head + 1u) & kMask, std::memory_order_release);
        return true;
    }

    [[nodiscard]] core::usize drain(std::span<T> out)
    {
        core::usize count = 0u;

        while (count < out.size() && pop(out[count]))
            ++count;
        return count;
    }

private:
    static constexpr core::usize kMask = Capacity - 1u;

    std::array<T, Capacity> _buffer{};
    std::atomic<core::usize> _head{0u};
    std::atomic<core::usize> _tail{0u};
};

using SeedU32Ring = SeedRing<core::u32, kSlots>;
using RingU32 = container::RingBuffer<core::u32, kSlots>;
using SeedMessageRing = SeedRing<Message, kSlots>;
using RingMessage = container::RingBuffer<Message, kSlots>;

template <typename Element> [[nodiscard]] Element makeElement(core::u32 sequence) noexcept
{
    if constexpr (std::is_same_v<Element, Message>)
        return Message{.sequence = sequence};
    else
        return sequence;
}

template <typename Element> [[nodiscard]] core::u32 sequenceOf(const Element &element) noexcept
{
    if constexpr (std::is_same_v<Element, Message>)
        return element.sequence;
    else
        return element;
}

/** @brief Sum of the sequence numbers 0 to kElementCount - 1, which a complete transfer adds up to. */
constexpr core::u64 kExpectedSum = static_cast<core::u64>(kElementCount) * (kElementCount - 1u) / 2u;

/** @brief One ring with what its consumer received in the last repetition. */
template <typename Ring> struct Transfer {
    Ring ring;               /**< The ring under test. */
    core::u64 received = 0u; /**< Sum of the sequence numbers the consumer received. */
};

template <typename Ring, typename Element> void consumeOneByOne(void *context) noexcept
{
    Transfer<Ring> &transfer = *static_cast<Transfer<Ring> *>(context);
    Element element{};
    core::u64 sum = 0u;

    for (core::u32 got = 0u; got < kElementCount;)
    {
        if (transfer.ring.pop(element))
        {
            sum += sequenceOf(element);
            ++got;
        }
    }
    transfer.received = sum;
}

template <typename Ring, typename Element> void produceOneByOne(Transfer<Ring> &transfer) noexcept
{
    for (core::u32 sequence = 0u; sequence < kElementCount;)
    {
        if (transfer.ring.push(makeElement<Element>(sequence)))
            ++sequence;
    }
}

template <typename Ring, typename Element> void consumeInBatches(void *context) noexcept
{
    Transfer<Ring> &transfer = *static_cast<Transfer<Ring> *>(context);
    core::u64 sum = 0u;

    for (core::u32 got = 0u; got < kElementCount;)
    {
        if constexpr (std::is_same_v<Ring, container::RingBuffer<Element, kSlots>>)
        {
            got += static_cast<core::u32>(
                transfer.ring.consume([&sum](Element &element) noexcept { sum += sequenceOf(element); }, kBatch));
        }
        else
        {
            std::array<Element, kBatch> batch{};
            const core::usize count = transfer.ring.drain(std::span<Element>(batch));
            for (core::usize i = 0u; i < count; ++i)
                sum += sequenceOf(batch[i]);
            got += static_cast<core::u32>(count);
        }
    }
    transfer.received = sum;
}

template <typename Ring, typename Element> void produceInBatches(Transfer<Ring> &transfer) noexcept
{
    std::array<Element, kBatch> batch{};

    for (core::u32 sequence = 0u; sequence < kElementCount;)
    {
        const core::u32 count = kElementCount - sequence < kBatch ? kElementCount - sequence : kBatch;
        for (core::u32 i = 0u; i < count; ++i)
            batch[i] = makeElement<Element>(sequence + i);
        if constexpr (std::is_same_v<Ring, container::RingBuffer<Element, kSlots>>)
        {
            sequence += static_cast<core::u32>(transfer.ring.pushBulk(std::span<const Element>(batch.data(), count)));
        }
        else
        {
            core::u32 pushed = 0u;
            while (pushed < count && transfer.ring.push(batch[pushed]))
                ++pushed;
            sequence += pushed;
        }
    }
}

/** @brief Two rings between the two cores, one each way, for a round trip. */
template <typename Ring> struct Echo {
    Ring there;         /**< Main thread to partner. */
    Ring back;          /**< Partner to main thread. */
    bool intact = true; /**< False once a round trip came back with the wrong sequence. */
};

template <typename Ring> void echoRoundTrips(void *context) noexcept
{
    Echo<Ring> &echo = *static_cast<Echo<Ring> *>(context);
    core::u32 value = 0u;

    for (core::u32 trip = 0u; trip < kRoundTrips; ++trip)
    {
        while (!echo.there.pop(value))
        {
        }
        while (!echo.back.push(value))
        {
        }
    }
}

template <typename Ring> void sendRoundTrips(Echo<Ring> &echo) noexcept
{
    core::u32 value = 0u;

    for (core::u32 trip = 0u; trip < kRoundTrips; ++trip)
    {
        while (!echo.there.push(trip))
        {
        }
        while (!echo.back.pop(value))
        {
        }
        echo.intact = echo.intact && value == trip;
    }
}

#if defined(__linux__)

[[nodiscard]] bool pinThisThread(int cpu) noexcept
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

[[nodiscard]] std::vector<int> siblingsOf(int cpu)
{
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    std::string list;
    std::vector<int> siblings;

    std::getline(file, list);
    for (std::size_t start = 0u; start < list.size();)
    {
        const std::size_t comma = list.find(',', start);
        const std::string range = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        const std::size_t dash = range.find('-');
        const int first = std::stoi(range.substr(0u, dash));
        const int last = dash == std::string::npos ? first : std::stoi(range.substr(dash + 1u));
        for (int sibling = first; sibling <= last; ++sibling)
            siblings.push_back(sibling);
        start = comma == std::string::npos ? list.size() : comma + 1u;
    }
    return siblings;
}

/** @brief The first core this process may run on, and the first other one that is not its SMT sibling. */
[[nodiscard]] std::optional<std::pair<int, int>> pickTwoPhysicalCores()
{
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        return std::nullopt;

    int first = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE && first < 0; ++cpu)
        if (CPU_ISSET(cpu, &allowed))
            first = cpu;
    if (first < 0)
        return std::nullopt;

    const std::vector<int> siblings = siblingsOf(first);
    for (int cpu = first + 1; cpu < CPU_SETSIZE; ++cpu)
    {
        if (!CPU_ISSET(cpu, &allowed))
            continue;
        bool isSibling = false;
        for (const int sibling : siblings)
            isSibling = isSibling || sibling == cpu;
        if (!isSibling)
            return std::pair{first, cpu};
    }
    return std::nullopt;
}

/**
 * @brief A thread pinned to one core that runs one job per request, spinning between them, so a
 *        repetition times the transfer and not a thread start.
 */
class PinnedPartner final {
public:
    explicit PinnedPartner(int cpu) : _thread([this, cpu] { serve(cpu); })
    {
        while (!_ready.load(std::memory_order_acquire))
            LPL_CPU_PAUSE();
    }

    ~PinnedPartner()
    {
        _request.store(kStop, std::memory_order_release);
        _thread.join();
    }

    PinnedPartner(const PinnedPartner &) = delete;
    PinnedPartner &operator=(const PinnedPartner &) = delete;

    void start(void (*job)(void *) noexcept, void *context) noexcept
    {
        _job = job;
        _context = context;
        _request.fetch_add(1u, std::memory_order_release);
    }

    void wait() const noexcept
    {
        const core::u32 request = _request.load(std::memory_order_relaxed);

        while (_done.load(std::memory_order_acquire) != request)
            LPL_CPU_PAUSE();
    }

    [[nodiscard]] bool pinned() const noexcept { return _pinned.load(std::memory_order_acquire); }

private:
    static constexpr core::u32 kStop = 0xFFFFFFFFu;

    void serve(int cpu) noexcept
    {
        core::u32 served = 0u;

        _pinned.store(pinThisThread(cpu), std::memory_order_relaxed);
        _ready.store(true, std::memory_order_release);
        for (;;)
        {
            core::u32 request = _request.load(std::memory_order_acquire);
            while (request == served)
            {
                LPL_CPU_PAUSE();
                request = _request.load(std::memory_order_acquire);
            }
            if (request == kStop)
                return;
            _job(_context);
            served = request;
            _done.store(served, std::memory_order_release);
        }
    }

    std::atomic<core::u32> _request{0u};
    std::atomic<core::u32> _done{0u};
    std::atomic<bool> _pinned{false};
    std::atomic<bool> _ready{false};
    void (*_job)(void *) noexcept = nullptr;
    void *_context = nullptr;
    std::thread _thread;
};

/** @brief Time of each repetition of one workload, per element or per round trip, for both rings. */
struct Comparison {
    const char *workload = "";             /**< What the two rings moved. */
    std::vector<core::f64> seedNs = {};    /**< Baseline ring, one entry per repetition. */
    std::vector<core::f64> currentNs = {}; /**< Current ring, one entry per repetition, paired with seedNs. */
};

template <typename Repetition> [[nodiscard]] core::f64 timeOnce(Repetition &repetition)
{
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    repetition();
    clobberMemory();
    return std::chrono::duration<core::f64, std::nano>(std::chrono::steady_clock::now() - start).count();
}

/**
 * @brief Times the two rings in pairs, one repetition each, swapping which goes first, so both meet
 *        the same placement of the two threads on the cores and the same load on the machine.
 */
template <typename Seed, typename Current>
[[nodiscard]] Comparison comparePaired(const char *workload, core::f64 perUnit, Seed &&seed, Current &&current)
{
    Comparison comparison{.workload = workload, .seedNs = {}, .currentNs = {}};

    for (core::u32 pair = 0u; pair < kWarmupPairs; ++pair)
    {
        seed();
        current();
    }
    for (core::u32 pair = 0u; pair < kPairs; ++pair)
    {
        if (pair % 2u == 0u)
        {
            comparison.seedNs.push_back(timeOnce(seed) / perUnit);
            comparison.currentNs.push_back(timeOnce(current) / perUnit);
        }
        else
        {
            comparison.currentNs.push_back(timeOnce(current) / perUnit);
            comparison.seedNs.push_back(timeOnce(seed) / perUnit);
        }
    }
    return comparison;
}

[[nodiscard]] core::f64 quantileOf(std::vector<core::f64> values, core::f64 quantile)
{
    std::sort(values.begin(), values.end());
    return values[static_cast<core::usize>(quantile * static_cast<core::f64>(values.size() - 1u))];
}

void printComparison(const Comparison &comparison)
{
    std::vector<core::f64> speedUps;

    for (core::usize pair = 0u; pair < comparison.seedNs.size(); ++pair)
        speedUps.push_back(comparison.seedNs[pair] / comparison.currentNs[pair]);
    std::printf("  %-34s %9.2f ns %9.2f ns %9.2fx  [%5.2fx .. %5.2fx]\n", comparison.workload,
                quantileOf(comparison.seedNs, 0.5), quantileOf(comparison.currentNs, 0.5), quantileOf(speedUps, 0.5),
                quantileOf(speedUps, 0.1), quantileOf(speedUps, 0.9));
}

template <typename Ring, typename Element>
[[nodiscard]] auto oneByOne(PinnedPartner &partner, Transfer<Ring> &transfer, bool &intact)
{
    return [&partner, &transfer, &intact] {
        partner.start(&consumeOneByOne<Ring, Element>, &transfer);
        produceOneByOne<Ring, Element>(transfer);
        partner.wait();
        intact = intact && transfer.received == kExpectedSum;
    };
}

template <typename Ring, typename Element>
[[nodiscard]] auto inBatches(PinnedPartner &partner, Transfer<Ring> &transfer, bool &intact)
{
    return [&partner, &transfer, &intact] {
        partner.start(&consumeInBatches<Ring, Element>, &transfer);
        produceInBatches<Ring, Element>(transfer);
        partner.wait();
        intact = intact && transfer.received == kExpectedSum;
    };
}

template <typename Ring> [[nodiscard]] auto roundTrips(PinnedPartner &partner, Echo<Ring> &echo)
{
    return [&partner, &echo] {
        partner.start(&echoRoundTrips<Ring>, &echo);
        sendRoundTrips(echo);
        partner.wait();
    };
}

void runPinned(int producerCore, int consumerCore)
{
    PinnedPartner partner(consumerCore);
    auto seedU32 = std::make_unique<Transfer<SeedU32Ring>>();
    auto ringU32 = std::make_unique<Transfer<RingU32>>();
    auto controlU32 = std::make_unique<Transfer<RingU32>>();
    auto seedMessages = std::make_unique<Transfer<SeedMessageRing>>();
    auto ringMessages = std::make_unique<Transfer<RingMessage>>();
    auto seedEcho = std::make_unique<Echo<SeedU32Ring>>();
    auto ringEcho = std::make_unique<Echo<RingU32>>();
    bool intact = true;

    std::printf("  producer on core %d, consumer on core %d, %zu slots, %u elements or %u round trips a "
                "repetition, %u pairs\n",
                producerCore, consumerCore, kSlots, kElementCount, kRoundTrips, kPairs);
    if (!partner.pinned())
        std::printf("  warning: the consumer could not be pinned\n");

    const std::vector<Comparison> comparisons{
        comparePaired("u32, one by one (per element)", kElementCount,
                      oneByOne<SeedU32Ring, core::u32>(partner, *seedU32, intact),
                      oneByOne<RingU32, core::u32>(partner, *ringU32, intact)),
        comparePaired("u32, batches of 64 (per element)", kElementCount,
                      inBatches<SeedU32Ring, core::u32>(partner, *seedU32, intact),
                      inBatches<RingU32, core::u32>(partner, *ringU32, intact)),
        comparePaired("64-byte message (per element)", kElementCount,
                      oneByOne<SeedMessageRing, Message>(partner, *seedMessages, intact),
                      oneByOne<RingMessage, Message>(partner, *ringMessages, intact)),
        comparePaired("u32 round trip (per trip)", kRoundTrips, roundTrips(partner, *seedEcho),
                      roundTrips(partner, *ringEcho)),
        comparePaired("control: RingBuffer against itself", kElementCount,
                      oneByOne<RingU32, core::u32>(partner, *controlU32, intact),
                      oneByOne<RingU32, core::u32>(partner, *ringU32, intact)),
    };

    if (!intact || !seedEcho->intact || !ringEcho->intact)
    {
        std::printf("  error: a transfer lost, duplicated or reordered an element; no result\n");
        return;
    }
    std::printf("\n  %-34s %12s %12s %10s  %s\n", "workload (median of the pairs)", "seed ring", "RingBuffer",
                "speed-up", "[10th .. 90th percentile]");
    for (const Comparison &comparison : comparisons)
        printComparison(comparison);
}

#endif

} // namespace

void runRingBenchmarks()
{
    section("SPSC ring: seed ring vs RingBuffer, producer and consumer on two physical cores");
#if defined(__linux__)
    const std::optional<std::pair<int, int>> cores = pickTwoPhysicalCores();
    cpu_set_t original;

    if (!cores)
    {
        std::printf("  skipped: this process may not run on two cores that are not SMT siblings\n");
        return;
    }
    CPU_ZERO(&original);
    pthread_getaffinity_np(pthread_self(), sizeof(original), &original);
    if (!pinThisThread(cores->first))
        std::printf("  warning: the producer could not be pinned\n");
    runPinned(cores->first, cores->second);
    pthread_setaffinity_np(pthread_self(), sizeof(original), &original);
#else
    std::printf("  skipped: pinning two threads is implemented on Linux only\n");
#endif
}

} // namespace lpl::bench
