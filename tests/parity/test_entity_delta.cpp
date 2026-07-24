/*
** LplPlugin — field-masked entity delta codec test (book §6.2.5)
**
** Proves the acked-baseline delta: an entity that stayed in range is encoded as
** only the fields that changed against the last state the server sent, a dormant
** entity costs id + one empty mask byte, and the receiver merges present fields
** onto the baseline it already holds (absent fields untouched). This is the
** "unchanged field costs one bit of absence" of the Quake III / DOOM III model.
*/

#include <lpl/net/protocol/EntityDelta.hpp>

#include <cstdio>

using namespace lpl;
using net::protocol::EntitySnapshot;

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

} // namespace

int main()
{
    std::printf("== entity delta codec ==\n");

    // ── A dormant entity: id + empty mask, nothing else ────────────────────── //
    {
        EntitySnapshot prev{};
        prev.id = 42;
        prev.px = 10.0f;
        EntitySnapshot cur = prev; // identical

        const core::u8 mask = net::protocol::computeFieldMask(prev, cur);
        check(mask == 0, "an unchanged entity yields an empty field mask");

        net::protocol::Bitstream w;
        net::protocol::writeEntityDelta(w, cur, mask);
        check(w.data().size() == 5, "a dormant entity costs 5 bytes (id + mask) vs 32 full");
    }

    // ── A single-axis move sends only that axis ────────────────────────────── //
    {
        EntitySnapshot prev{};
        prev.id = 7;
        prev.px = 1.0f;
        prev.py = 2.0f;
        prev.pz = 3.0f;
        EntitySnapshot cur = prev;
        cur.px = 1.5f; // only X moved

        const core::u8 mask = net::protocol::computeFieldMask(prev, cur);
        check(mask == net::protocol::FieldPosX, "only the moved axis is flagged");

        net::protocol::Bitstream w;
        net::protocol::writeEntityDelta(w, cur, mask);
        check(w.data().size() == 5 + 4, "a one-axis move costs id + mask + one float = 9 bytes");

        // Decode onto the receiver's stale baseline (== prev).
        net::protocol::Bitstream r{w.data(), w.bitsWritten()};
        EntitySnapshot held = prev;
        core::u32 id = 0;
        core::u8 got = 0xFF;
        check(net::protocol::readEntityDelta(r, held, id, got).has_value(), "delta decodes");
        check(id == 7 && got == net::protocol::FieldPosX, "decoded id and mask match");
        check(held.px == 1.5f, "the changed axis is applied");
        check(held.py == 2.0f && held.pz == 3.0f, "untouched axes keep the baseline value");
    }

    // ── A full keyframe (mask = all) rebuilds the whole entity ─────────────── //
    {
        EntitySnapshot cur{};
        cur.id = 99;
        cur.px = -4.0f;
        cur.py = 5.0f;
        cur.pz = 6.0f;
        cur.sx = 2.0f;
        cur.sy = 2.0f;
        cur.sz = 2.0f;
        cur.hp = 55;

        net::protocol::Bitstream w;
        net::protocol::writeEntityDelta(w, cur, net::protocol::FieldAll);
        check(w.data().size() == 5 + 7 * 4, "a keyframe carries id + mask + 7 fields = 33 bytes");

        net::protocol::Bitstream r{w.data(), w.bitsWritten()};
        EntitySnapshot held{}; // no prior belief at all
        core::u32 id = 0;
        core::u8 got = 0;
        check(net::protocol::readEntityDelta(r, held, id, got).has_value(), "keyframe decodes");
        check(got == net::protocol::FieldAll, "keyframe mask is all fields");
        check(held.px == -4.0f && held.py == 5.0f && held.pz == 6.0f, "keyframe restores position from nothing");
        check(held.sx == 2.0f && held.hp == 55, "keyframe restores size and hp");
    }

    // ── Multiple fields, chained deltas converge to the source ─────────────── //
    {
        EntitySnapshot source{};
        source.id = 3;
        EntitySnapshot receiver{};
        receiver.id = 3;

        // Frame 1: move + damage.
        EntitySnapshot f1 = source;
        f1.px = 8.0f;
        f1.hp = 90;
        core::u8 m1 = net::protocol::computeFieldMask(source, f1);
        net::protocol::Bitstream w1;
        net::protocol::writeEntityDelta(w1, f1, m1);
        net::protocol::Bitstream r1{w1.data(), w1.bitsWritten()};
        core::u32 id;
        core::u8 got;
        (void) net::protocol::readEntityDelta(r1, receiver, id, got);

        // Frame 2: grow only.
        EntitySnapshot f2 = f1;
        f2.sy = 3.0f;
        core::u8 m2 = net::protocol::computeFieldMask(f1, f2);
        net::protocol::Bitstream w2;
        net::protocol::writeEntityDelta(w2, f2, m2);
        net::protocol::Bitstream r2{w2.data(), w2.bitsWritten()};
        (void) net::protocol::readEntityDelta(r2, receiver, id, got);

        check(receiver.px == 8.0f && receiver.hp == 90 && receiver.sy == 3.0f,
              "chained field deltas converge to the source state");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
