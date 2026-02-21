// File: test_metrics.cpp
// Description: Tests unitaires pour SignalMetrics (Schumacher, RMS, baseline)
// Build: g++ -std=c++20 -I../include -I../../shared -o test_metrics test_metrics.cpp && ./test_metrics
// Auteur: MasterLaplace

#include "SignalMetrics.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// ─── Utilitaire ──────────────────────────────────────────────────────────────

static bool near(float a, float b, float eps = 1e-4f) { return std::abs(a - b) < eps; }

// ─── Tests integrale ──────────────────────────────────────────────────────────

static void test_integrale_zero()
{
    std::vector<float> psd(128, 0.0f);
    assert(SignalMetrics::integrale(40u, 70u, psd) == 0.0f);
    printf("[PASS] integrale(PSD=0) = 0\n");
}

static void test_integrale_uniform()
{
    // PSD = 1.0 sur tous les bins
    // integrale(40, 71) = bins 40..71 inclus = 32 bins × 1.0 = 32.0
    std::vector<float> psd(128, 1.0f);
    float r = SignalMetrics::integrale(40u, 71u, psd);
    assert(near(r, 32.0f));
    printf("[PASS] integrale(PSD=1, bins 40..71) = %.4f  (expected 32.0)\n", r);
}

static void test_integrale_bounds_check()
{
    // upper_b > taille du vecteur → doit clamp sans crash
    std::vector<float> psd(50, 1.0f);
    float r = SignalMetrics::integrale(40u, 200u, psd); // upper_b clampé à 49
    assert(near(r, 10.0f));                             // bins 40..49 = 10 bins
    printf("[PASS] integrale(upper_b hors bornes) = %.4f  (expected 10.0)\n", r);
}

// ─── Tests schumacher ─────────────────────────────────────────────────────────

static void test_schumacher_uniform_8ch()
{
    // 8 canaux avec PSD uniforme = 1.0
    // integrale(40, 71) = 32 pour chaque canal → moyenne = 32.0
    std::vector<std::vector<float>> channels(8, std::vector<float>(128, 1.0f));
    float r = SignalMetrics::schumacher(channels, 40u, 71u);
    assert(near(r, 32.0f));
    printf("[PASS] schumacher(PSD=1, 8ch) = %.4f  (expected 32.0)\n", r);
}

static void test_schumacher_single_peak()
{
    // Canal 0 : pic = 4.0 au bin 55 (~53 Hz)
    // Canaux 1–3 : zéro partout
    // → R(t) = 4.0 / 4 canaux = 1.0
    std::vector<std::vector<float>> channels(4, std::vector<float>(128, 0.0f));
    channels[0][55] = 4.0f;
    float r = SignalMetrics::schumacher(channels, 40u, 71u);
    assert(near(r, 1.0f));
    printf("[PASS] schumacher(pic unique ch0=4.0, 4ch) = %.4f  (expected 1.0)\n", r);
}

static void test_schumacher_empty()
{
    std::vector<std::vector<float>> channels;
    float r = SignalMetrics::schumacher(channels);
    assert(r == 0.0f);
    printf("[PASS] schumacher(empty) = 0\n");
}

// ─── Tests hz_to_bin ──────────────────────────────────────────────────────────

static void test_hz_to_bin()
{
    // fs=250 Hz, N=256 → résolution = 250/256 ≈ 0.977 Hz/bin
    // 40 Hz → 40 × 256/250 = 40.96 → bin 40
    // 70 Hz → 70 × 256/250 = 71.68 → bin 71
    uint16_t b40 = SignalMetrics::hz_to_bin(40.0f, 250.0f, 256);
    uint16_t b70 = SignalMetrics::hz_to_bin(70.0f, 250.0f, 256);
    assert(b40 == 40u);
    assert(b70 == 71u);
    printf("[PASS] hz_to_bin(40Hz)=%u  hz_to_bin(70Hz)=%u\n", b40, b70);
}

// ─── Tests sliding_window_rms ─────────────────────────────────────────────────

static void test_rms_constant()
{
    // Signal constant = 3.0 → RMS = 3.0
    std::vector<float> data(64, 3.0f);
    float rms = SignalMetrics::sliding_window_rms(data, 32u);
    assert(near(rms, 3.0f));
    printf("[PASS] sliding_window_rms(const=3.0) = %.4f\n", rms);
}

static void test_rms_alternating()
{
    // Signal alternant +1/-1 → RMS = 1.0
    std::vector<float> data(64);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    float rms = SignalMetrics::sliding_window_rms(data, 64u);
    assert(near(rms, 1.0f));
    printf("[PASS] sliding_window_rms(alternant ±1) = %.4f\n", rms);
}

static void test_rms_window_too_large()
{
    std::vector<float> data(10, 1.0f);
    float rms = SignalMetrics::sliding_window_rms(data, 20u);
    assert(rms == 0.0f);
    printf("[PASS] sliding_window_rms(window > data) = 0\n");
}

// ─── Tests compute_baseline ──────────────────────────────────────────────────

static void test_baseline_known()
{
    // {1, 2, 3, 4, 5} → mean=3, var=(4+1+0+1+4)/5=2 → std=sqrt(2)
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto bl = SignalMetrics::compute_baseline(data);
    assert(near(bl.mean, 3.0f));
    assert(near(bl.std_dev, std::sqrt(2.0f)));
    printf("[PASS] compute_baseline: mean=%.4f  std=%.6f  (expected 3.0 / %.6f)\n", bl.mean, bl.std_dev,
           std::sqrt(2.0f));
}

static void test_baseline_empty()
{
    std::vector<float> data;
    auto bl = SignalMetrics::compute_baseline(data);
    assert(bl.mean == 0.0f && bl.std_dev == 0.0f);
    printf("[PASS] compute_baseline(empty) = {0, 0}\n");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main()
{
    printf("=== SignalMetrics — tests unitaires ===\n\n");

    printf("-- integrale --\n");
    test_integrale_zero();
    test_integrale_uniform();
    test_integrale_bounds_check();

    printf("\n-- schumacher R(t) --\n");
    test_schumacher_uniform_8ch();
    test_schumacher_single_peak();
    test_schumacher_empty();

    printf("\n-- hz_to_bin --\n");
    test_hz_to_bin();

    printf("\n-- sliding_window_rms --\n");
    test_rms_constant();
    test_rms_alternating();
    test_rms_window_too_large();

    printf("\n-- compute_baseline --\n");
    test_baseline_known();
    test_baseline_empty();

    printf("\n[OK] Tous les tests sont passes.\n");
    return 0;
}
