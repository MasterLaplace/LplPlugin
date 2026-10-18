#include <lpl/net/protocol/EntityDelta.hpp>

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

/**
 * @brief A dormant entity: id + empty mask, nothing else.
 */
void checkDormantEntityCostsIdAndEmptyMask()
{
    lpl::net::protocol::EntitySnapshot prev{};
    prev.id = 42;
    prev.px = 10.0f;
    lpl::net::protocol::EntitySnapshot cur = prev;

    const lpl::core::u8 mask = lpl::net::protocol::computeFieldMask(prev, cur);
    check(mask == 0, "an unchanged entity yields an empty field mask");

    lpl::net::protocol::Bitstream w;
    lpl::net::protocol::writeEntityDelta(w, cur, mask);
    check(w.data().size() == 5, "a dormant entity costs 5 bytes (id + mask) vs 32 full");
}

/**
 * @brief A single-axis move sends only that axis.
 *
 * @details The delta is decoded onto the receiver's stale baseline, the previous state.
 */
void checkSingleAxisMoveSendsOnlyThatAxis()
{
    lpl::net::protocol::EntitySnapshot prev{};
    prev.id = 7;
    prev.px = 1.0f;
    prev.py = 2.0f;
    prev.pz = 3.0f;
    lpl::net::protocol::EntitySnapshot cur = prev;
    cur.px = 1.5f;

    const lpl::core::u8 mask = lpl::net::protocol::computeFieldMask(prev, cur);
    check(mask == lpl::net::protocol::FieldPosX, "only the moved axis is flagged");

    lpl::net::protocol::Bitstream w;
    lpl::net::protocol::writeEntityDelta(w, cur, mask);
    check(w.data().size() == 5 + 4, "a one-axis move costs id + mask + one float = 9 bytes");

    lpl::net::protocol::Bitstream r{w.data(), w.bitsWritten()};
    lpl::net::protocol::EntitySnapshot held = prev;
    lpl::core::u32 id = 0;
    lpl::core::u8 got = 0xFF;
    check(lpl::net::protocol::readEntityDelta(r, held, id, got).has_value(), "delta decodes");
    check(id == 7 && got == lpl::net::protocol::FieldPosX, "decoded id and mask match");
    check(held.px == 1.5f, "the changed axis is applied");
    check(held.py == 2.0f && held.pz == 3.0f, "untouched axes keep the baseline value");
}

/**
 * @brief A full keyframe (mask = all) rebuilds the whole entity.
 *
 * @details The receiver decodes it with no prior belief at all.
 */
void checkFullKeyframeRebuildsTheWholeEntity()
{
    lpl::net::protocol::EntitySnapshot cur{};
    cur.id = 99;
    cur.px = -4.0f;
    cur.py = 5.0f;
    cur.pz = 6.0f;
    cur.sx = 2.0f;
    cur.sy = 2.0f;
    cur.sz = 2.0f;
    cur.hp = 55;

    lpl::net::protocol::Bitstream w;
    lpl::net::protocol::writeEntityDelta(w, cur, lpl::net::protocol::FieldAll);
    check(w.data().size() == 5 + 7 * 4, "a keyframe carries id + mask + 7 fields = 33 bytes");

    lpl::net::protocol::Bitstream r{w.data(), w.bitsWritten()};
    lpl::net::protocol::EntitySnapshot held{};
    lpl::core::u32 id = 0;
    lpl::core::u8 got = 0;
    check(lpl::net::protocol::readEntityDelta(r, held, id, got).has_value(), "keyframe decodes");
    check(got == lpl::net::protocol::FieldAll, "keyframe mask is all fields");
    check(held.px == -4.0f && held.py == 5.0f && held.pz == 6.0f, "keyframe restores position from nothing");
    check(held.sx == 2.0f && held.hp == 55, "keyframe restores size and hp");
}

/**
 * @brief Multiple fields, chained deltas converge to the source.
 *
 * @details Frame 1 moves and damages the entity, frame 2 only grows it; the receiver applies both deltas in
 *          turn.
 */
void checkChainedDeltasConvergeToTheSource()
{
    lpl::net::protocol::EntitySnapshot source{};
    source.id = 3;
    lpl::net::protocol::EntitySnapshot receiver{};
    receiver.id = 3;

    lpl::net::protocol::EntitySnapshot f1 = source;
    f1.px = 8.0f;
    f1.hp = 90;
    lpl::core::u8 m1 = lpl::net::protocol::computeFieldMask(source, f1);
    lpl::net::protocol::Bitstream w1;
    lpl::net::protocol::writeEntityDelta(w1, f1, m1);
    lpl::net::protocol::Bitstream r1{w1.data(), w1.bitsWritten()};
    lpl::core::u32 id;
    lpl::core::u8 got;
    (void) lpl::net::protocol::readEntityDelta(r1, receiver, id, got);

    lpl::net::protocol::EntitySnapshot f2 = f1;
    f2.sy = 3.0f;
    lpl::core::u8 m2 = lpl::net::protocol::computeFieldMask(f1, f2);
    lpl::net::protocol::Bitstream w2;
    lpl::net::protocol::writeEntityDelta(w2, f2, m2);
    lpl::net::protocol::Bitstream r2{w2.data(), w2.bitsWritten()};
    (void) lpl::net::protocol::readEntityDelta(r2, receiver, id, got);

    check(receiver.px == 8.0f && receiver.hp == 90 && receiver.sy == 3.0f,
          "chained field deltas converge to the source state");
}

/**
 * @brief Quantized far-ring delta: coarser position, fewer bytes (§6.2.6).
 *
 * @details Positions take 16 bits, 2 bytes per axis. A keyframe then costs id 4 + mask 1 + 3*2 (pos 16-bit)
 *          + 3*4 (size full) + 4 (hp) = 27 bytes, vs 33 full.
 */
void checkQuantizedFarRingDeltaIsSmaller()
{
    lpl::net::protocol::EntitySnapshot cur{};
    cur.id = 5;
    cur.px = 123.4f;
    cur.py = -50.0f;
    cur.pz = 999.9f;
    cur.sx = 2.0f;
    cur.sy = 2.0f;
    cur.sz = 2.0f;
    cur.hp = 70;

    const float extent = 1024.0f;
    const lpl::core::u32 bits = 16;

    lpl::net::protocol::Bitstream w;
    lpl::net::protocol::writeEntityDeltaQuantized(w, cur, lpl::net::protocol::FieldAll, extent, bits);
    check(w.data().size() == 27, "a quantized keyframe is smaller than the full-float one (27 vs 33)");

    lpl::net::protocol::Bitstream r{w.data(), w.bitsWritten()};
    lpl::net::protocol::EntitySnapshot held{};
    lpl::core::u32 id = 0;
    lpl::core::u8 mask = 0;
    check(lpl::net::protocol::readEntityDeltaQuantized(r, held, id, mask, extent, bits).has_value(),
          "quantized entity decodes");

    const float step = 2.0f * extent / static_cast<float>((1u << bits) - 1u);
    check(std::fabs(held.px - cur.px) <= step && std::fabs(held.pz - cur.pz) <= step,
          "quantized position is within one quantization step");
    check(held.sx == 2.0f && held.sz == 2.0f && held.hp == 70, "size and hp stay exact (full precision)");
}

} // namespace

/**
 * @brief Field-masked entity delta codec test (book §6.2.5).
 *
 * @details Proves the acked-baseline delta: an entity that stayed in range is encoded as only the fields that
 *          changed against the last state the server sent, a dormant entity costs id + one empty mask byte,
 *          and the receiver merges present fields onto the baseline it already holds (absent fields
 *          untouched). This is the "unchanged field costs one bit of absence" of the Quake III / DOOM III
 *          model.
 */
int main()
{
    std::printf("== entity delta codec ==\n");

    checkDormantEntityCostsIdAndEmptyMask();
    checkSingleAxisMoveSendsOnlyThatAxis();
    checkFullKeyframeRebuildsTheWholeEntity();
    checkChainedDeltasConvergeToTheSource();
    checkQuantizedFarRingDeltaIsSmaller();

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
