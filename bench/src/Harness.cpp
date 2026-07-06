/**
 * @file Harness.cpp
 * @brief Non-template implementation of the benchmark harness: sample
 *        reduction, formatting, and reporting.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-06
 * @copyright MIT License
 */

#include <lpl/bench/Harness.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace lpl::bench {

std::string formatDuration(core::f64 ns)
{
    const char *unit;
    core::f64 v;
    if (ns < 1e3)
    {
        v = ns;
        unit = "ns";
    }
    else if (ns < 1e6)
    {
        v = ns / 1e3;
        unit = "us";
    }
    else if (ns < 1e9)
    {
        v = ns / 1e6;
        unit = "ms";
    }
    else
    {
        v = ns / 1e9;
        unit = "s";
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g %s", v, unit);
    return std::string(buf);
}

const char *frameRateVerdict(core::f64 msPerFrame)
{
    const core::f64 fps = 1000.0 / (msPerFrame > 0.0 ? msPerFrame : 1.0);
    if (fps >= 60.0)
        return "REAL-TIME  (>=60 fps)";
    if (fps >= 30.0)
        return "PLAYABLE   (>=30 fps)";
    return "TOO SLOW   (<30 fps) ";
}

void printLegend()
{
    std::printf("Legend: median  ±CV%%  [min … p99]  n=samples  (lower is better)\n\n");
}

void section(const char *title)
{
    std::printf("\n  --- %s ---\n", title);
}

Result report(const char *label, std::vector<core::f64> &samplesNs)
{
    std::sort(samplesNs.begin(), samplesNs.end());

    const std::size_t n = samplesNs.size();
    Result r;
    r.samples = static_cast<core::u32>(n);
    r.minNs = samplesNs.front();
    r.medianNs = samplesNs[n / 2];
    const std::size_t p99Index = (n * 99) / 100 < n ? (n * 99) / 100 : n - 1;
    r.p99Ns = samplesNs[p99Index];

    core::f64 sum = 0.0;
    for (core::f64 s : samplesNs)
        sum += s;
    r.meanNs = sum / static_cast<core::f64>(n);

    core::f64 var = 0.0;
    for (core::f64 s : samplesNs)
    {
        const core::f64 d = s - r.meanNs;
        var += d * d;
    }
    r.stddevNs = n > 1 ? std::sqrt(var / static_cast<core::f64>(n - 1)) : 0.0;

    const core::f64 cv = r.meanNs > 0.0 ? (r.stddevNs / r.meanNs) * 100.0 : 0.0;
    std::printf("  %-40s %11s  ±%4.1f%%  [min %-10s p99 %-10s] n=%u\n", label, formatDuration(r.medianNs).c_str(), cv,
                formatDuration(r.minNs).c_str(), formatDuration(r.p99Ns).c_str(), r.samples);
    return r;
}

} // namespace lpl::bench
