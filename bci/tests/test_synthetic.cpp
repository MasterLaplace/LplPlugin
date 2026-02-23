// File: test_synthetic.cpp
// Description: Tests d'intégration pour le pipeline synthetic → FFT → NeuralState → Calibration
// Vérifie que le générateur synthétique produit des données réalistes et que la
// pipeline complète fonctionne sans matériel.
//
// Build: g++ -std=c++20 -I../include -I../../shared -o test_synthetic test_synthetic.cpp && ./test_synthetic
// Auteur: MasterLaplace

#include "Calibration.hpp"
#include "NeuralMetrics.hpp"
#include "SyntheticSource.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

// ─── Utilitaires ──────────────────────────────────────────────────────────────

static bool near(float a, float b, float eps = 1e-3f) { return std::abs(a - b) < eps; }
static bool in_range(float v, float lo, float hi) { return v >= lo && v <= hi; }

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS : SyntheticGenerator
// ═══════════════════════════════════════════════════════════════════════════════

static void test_generator_deterministic()
{
    SyntheticGenerator gen1(42);
    SyntheticGenerator gen2(42);

    auto s1 = gen1.generate(256);
    auto s2 = gen2.generate(256);

    for (size_t t = 0; t < 256; ++t)
        for (size_t ch = 0; ch < SYNTH_CHANNELS; ++ch)
            assert(s1[t][ch] == s2[t][ch]);

    printf("[PASS] SyntheticGenerator : graine identique → sortie identique\n");
}

static void test_generator_different_seeds()
{
    SyntheticGenerator gen1(42);
    SyntheticGenerator gen2(99);

    auto s1 = gen1.generate(256);
    auto s2 = gen2.generate(256);

    bool different = false;
    for (size_t t = 0; t < 256 && !different; ++t)
        for (size_t ch = 0; ch < SYNTH_CHANNELS && !different; ++ch)
            if (s1[t][ch] != s2[t][ch])
                different = true;

    assert(different);
    printf("[PASS] SyntheticGenerator : graines différentes → sortie différente\n");
}

static void test_generator_amplitude_range()
{
    SyntheticGenerator gen(42);
    auto samples = gen.generate(2048);

    float minVal = 1e9f, maxVal = -1e9f;
    for (const auto &row : samples)
        for (float v : row)
        {
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }

    // Avec alpha=15µV, beta=8µV, gamma=3µV, noise=2µV, blink=200µV
    // Le max attendu est ~230µV (avec clignement), le min ~-30µV
    assert(maxVal > 5.0f);   // Signal non nul
    assert(minVal < -5.0f);  // Signal bipolaire
    assert(maxVal < 500.0f); // Pas d'explosion

    printf("[PASS] SyntheticGenerator : amplitude réaliste (min=%.1f, max=%.1f µV)\n", minVal, maxVal);
}

static void test_generator_channel_count()
{
    SyntheticGenerator gen(42);
    auto samples = gen.generate(10);

    assert(samples.size() == 10);
    for (const auto &row : samples)
        assert(row.size() == SYNTH_CHANNELS);

    printf("[PASS] SyntheticGenerator : 8 canaux x N samples\n");
}

static void test_generator_reset()
{
    SyntheticGenerator gen(42);
    auto s1 = gen.generate(100);
    gen.reset(42);
    auto s2 = gen.generate(100);

    for (size_t t = 0; t < 100; ++t)
        for (size_t ch = 0; ch < SYNTH_CHANNELS; ++ch)
            assert(s1[t][ch] == s2[t][ch]);

    printf("[PASS] SyntheticGenerator : reset(seed) reproduit la même séquence\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS : SyntheticSource (pipeline complète)
// ═══════════════════════════════════════════════════════════════════════════════

static void test_source_init()
{
    SyntheticSource src(42, false);
    assert(src.init());
    assert(src.mode() == BciMode::Synthetic);
    src.stop();
    printf("[PASS] SyntheticSource : init() réussit\n");
}

static void test_source_produces_neural_state()
{
    SyntheticSource src(42, false);
    [[maybe_unused]] bool ok = src.init();

    NeuralState state;

    // Faire plusieurs updates pour remplir le buffer FFT
    for (int i = 0; i < 20; ++i)
        src.update(state);

    // Après 20 × UPDATE_INTERVAL (32) = 640 samples, la FFT a tourné
    assert(state.alphaPower > 0.0f);
    assert(state.betaPower > 0.0f);
    assert(in_range(state.concentration, 0.0f, 1.0f));
    assert(state.schumacherR >= 0.0f);

    src.stop();
    printf("[PASS] SyntheticSource : alpha=%.4f, beta=%.4f, R(t)=%.4f, conc=%.4f\n", state.alphaPower, state.betaPower,
           state.schumacherR, state.concentration);
}

static void test_source_produces_per_channel()
{
    SyntheticSource src(42, false);
    [[maybe_unused]] bool ok = src.init();

    NeuralState state;
    for (int i = 0; i < 20; ++i)
        src.update(state);

    // Chaque canal doit avoir de la puissance alpha et beta
    for (size_t ch = 0; ch < BCI_CHANNELS; ++ch)
    {
        assert(state.channelAlpha[ch] >= 0.0f);
        assert(state.channelBeta[ch] >= 0.0f);
    }

    src.stop();
    printf("[PASS] SyntheticSource : puissance par canal valide (8 canaux)\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS : NeuralMetrics depuis SyntheticSource
// ═══════════════════════════════════════════════════════════════════════════════

static void test_metrics_from_synthetic()
{
    SyntheticSource src(42, false);
    [[maybe_unused]] bool ok = src.init();

    NeuralState state;
    for (int i = 0; i < 20; ++i)
        src.update(state);

    // Sans baseline
    auto m1 = NeuralMetrics::from_state(state, 0.0f);
    assert(in_range(m1.concentration, 0.0f, 1.0f));
    assert(m1.signal_quality == 1.0f || m1.signal_quality == 0.0f);

    // Avec baseline simulée
    auto m2 = NeuralMetrics::from_state(state, state.schumacherR * 1.5f);
    assert(in_range(m2.muscle_tension, 0.0f, 1.0f));
    assert(in_range(m2.stability, 0.0f, 1.0f));

    src.stop();
    printf("[PASS] NeuralMetrics : muscle_tension=%.4f, stability=%.4f, conc=%.4f\n", m2.muscle_tension, m2.stability,
           m2.concentration);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS : Calibration
// ═══════════════════════════════════════════════════════════════════════════════

static void test_calibration_state_machine()
{
    Calibration calib(0.5f, 5); // 0.5 sec, min 5 samples

    assert(calib.state() == CalibState::Idle);
    assert(near(calib.progress(), 0.0f));

    calib.start();
    assert(calib.state() == CalibState::Calibrating);

    printf("[PASS] Calibration : transitions Idle → Calibrating\n");
}

static void test_calibration_with_synthetic()
{
    SyntheticSource src(42, false);
    [[maybe_unused]] bool ok = src.init();

    // Calibration très courte pour les tests (0.1s, min 3 samples)
    Calibration calib(0.1f, 3);
    calib.start();

    NeuralState state;
    int iterations = 0;

    // Boucle jusqu'à Ready (ou timeout)
    while (calib.state() != CalibState::Ready && iterations < 1000)
    {
        src.update(state);
        calib.tick(state);
        iterations++;
    }

    assert(calib.state() == CalibState::Ready);
    assert(calib.baseline().mean >= 0.0f);
    assert(calib.sampleCount() >= 3);
    assert(near(calib.progress(), 1.0f));

    // Les metrics avec baseline calibrée doivent être cohérentes
    auto metrics = calib.computeMetrics(state);
    assert(in_range(metrics.muscle_tension, 0.0f, 2.0f)); // Peut dépasser 1 si R(t) > baseline
    assert(in_range(metrics.stability, 0.0f, 1.0f));

    src.stop();
    printf("[PASS] Calibration : baseline_R=%.4f ± %.4f (%zu samples, %d iterations)\n", calib.baseline().mean,
           calib.baseline().std_dev, calib.sampleCount(), iterations);
}

static void test_calibration_force_complete()
{
    Calibration calib(100.0f, 3); // Long duration, but we'll force it
    calib.start();

    // Inject fake R(t) values
    NeuralState state;
    state.schumacherR = 0.5f;
    for (int i = 0; i < 10; ++i)
        calib.tick(state);

    assert(calib.state() == CalibState::Calibrating);

    calib.forceComplete();
    assert(calib.state() == CalibState::Ready);
    assert(near(calib.baseline().mean, 0.5f, 0.01f));

    printf("[PASS] Calibration : forceComplete() fonctionne (baseline=%.4f)\n", calib.baseline().mean);
}

static void test_calibration_reset()
{
    Calibration calib(0.1f, 3);
    calib.start();

    NeuralState state;
    state.schumacherR = 1.0f;
    for (int i = 0; i < 10; ++i)
        calib.tick(state);

    calib.forceComplete();
    assert(calib.state() == CalibState::Ready);

    calib.reset();
    assert(calib.state() == CalibState::Idle);
    assert(calib.sampleCount() == 0);

    printf("[PASS] Calibration : reset() → retour Idle\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS : BciMode parsing
// ═══════════════════════════════════════════════════════════════════════════════

static void test_bci_mode_parsing()
{
    assert(bci_mode_from_string("serial") == BciMode::Serial);
    assert(bci_mode_from_string("synthetic") == BciMode::Synthetic);
    assert(bci_mode_from_string("synth") == BciMode::Synthetic);
    assert(bci_mode_from_string("lsl") == BciMode::Lsl);
    assert(bci_mode_from_string("csv") == BciMode::CsvReplay);
    assert(bci_mode_from_string("replay") == BciMode::CsvReplay);
    assert(bci_mode_from_string("unknown") == BciMode::Serial); // Default

    printf("[PASS] bci_mode_from_string : tous les modes reconnus\n");
}

static void test_bci_mode_names()
{
    assert(std::string(bci_mode_name(BciMode::Serial)) == "Serial (OpenBCI)");
    assert(std::string(bci_mode_name(BciMode::Synthetic)) == "Synthetic");
    assert(std::string(bci_mode_name(BciMode::Lsl)) == "LSL Inlet");
    assert(std::string(bci_mode_name(BciMode::CsvReplay)) == "CSV Replay");

    printf("[PASS] bci_mode_name : noms corrects\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main()
{
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Tests d'intégration : SyntheticGenerator + Pipeline BCI\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    // SyntheticGenerator
    test_generator_deterministic();
    test_generator_different_seeds();
    test_generator_amplitude_range();
    test_generator_channel_count();
    test_generator_reset();

    printf("\n");

    // SyntheticSource (pipeline FFT)
    test_source_init();
    test_source_produces_neural_state();
    test_source_produces_per_channel();

    printf("\n");

    // NeuralMetrics
    test_metrics_from_synthetic();

    printf("\n");

    // Calibration
    test_calibration_state_machine();
    test_calibration_with_synthetic();
    test_calibration_force_complete();
    test_calibration_reset();

    printf("\n");

    // BciMode parsing
    test_bci_mode_parsing();
    test_bci_mode_names();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  TOUS LES TESTS PASSENT ✓\n");
    printf("═══════════════════════════════════════════════════════════\n");

    return 0;
}
