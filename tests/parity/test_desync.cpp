/*
** LplPlugin — desync detection + diagnostic capture test (book §6.4)
**
** Proves the server catches a client whose simulation diverged from the
** authoritative one, and captures the divergence for post-mortem: the tick, both
** digests, and who reported it (§6.4.2). A client reports a digest for a past
** tick; the server looks that tick up in its history and, when the digests
** disagree, records a DesyncReport — the hook a diagnostic tool hangs off.
*/

#include <lpl/engine/Server.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/net/Endpoint.hpp>

#    include <cstdio>

using namespace lpl;

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
    std::printf("== desync detection + diagnostics ==\n");

    auto config = engine::Config::Builder{}
                      .serverMode(true)
                      .tickRate(60)
                      .serverPort(45996)
                      .replaySnapshotInterval(1) // keep a snapshot each tick for post-mortem
                      .build();
    engine::Server server{config};
    check(server.init().has_value(), "server opens its socket");

    const auto world = server.addWorld(lpl::pmr::make_unique<engine::World>());
    check(world != engine::Server::kInvalidWorldId, "an instance is hosted");

    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 6; ++i)
        server.tick(dt);

    // A client reports a digest for one of its past ticks that disagrees with what
    // the server held for that tick. Scan back for the first in-history tick that
    // a deliberately wrong digest diverges from (the empty world hashes to a fixed
    // value each tick, so any different digest diverges).
    const core::u64 wrongDigest = 0xDEADBEEFCAFEBABEull;
    core::u64 targetTick = 0;
    bool foundDivergent = false;
    for (core::u64 t = server.currentTick(); t-- > 0;)
    {
        if (server.checkClientHash(world, t, wrongDigest) == engine::Server::DesyncVerdict::Diverged)
        {
            targetTick = t;
            foundDivergent = true;
            break;
        }
    }
    check(foundDivergent, "a wrong digest for an in-history tick is detected as a divergence");

    // No divergence has been *captured* yet — detection via checkClientHash is
    // pure; capture happens only when a report flows through the tick.
    engine::Server::DesyncReport before{};
    check(!server.lastDesyncReport(before), "nothing is captured until a report is actually processed");

    // Push the report into the instance's queue, exactly where a decoded
    // StateHashReport packet lands, then tick so the server consumes it.
    const auto reporter = net::Endpoint::fromOctets(127, 0, 0, 1, 51000);
    engine::StateHashReportEvent report{};
    report.source = reporter;
    report.tick = targetTick;
    report.digest = wrongDigest;
    server.queues(world)->stateHashReports.push(std::move(report));

    const core::u64 desyncsBefore = server.desyncCount();
    server.tick(dt);

    check(server.desyncCount() == desyncsBefore + 1, "the divergence is counted");

    engine::Server::DesyncReport captured{};
    check(server.lastDesyncReport(captured), "the divergence is captured for post-mortem");
    check(captured.tick == targetTick, "the captured report names the divergent tick");
    check(captured.clientDigest == wrongDigest, "it records the client's reported digest");
    check(captured.serverDigest != captured.clientDigest, "server and client digests differ — that IS the divergence");
    check(captured.instance == world, "it records which instance diverged");
    check(captured.source == reporter, "it records who reported it");

    // A matching report must NOT be flagged: feed the server its own digest for a
    // tick and confirm it agrees (no false positive).
    core::u64 matchTick = 0;
    bool foundMatch = false;
    const core::u64 serverNow = server.stateHash(world); // digest of the latest folded tick
    for (core::u64 t = server.currentTick(); t-- > 0;)
    {
        if (server.checkClientHash(world, t, serverNow) == engine::Server::DesyncVerdict::Match)
        {
            matchTick = t;
            foundMatch = true;
            break;
        }
    }
    check(foundMatch, "the server agrees with a client that reports the correct digest (no false positive)");
    (void) matchTick;

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#else

#    include <cstdio>

int main()
{
    std::printf("desync test skipped: built without LPL_HAS_NET\n");
    return 0;
}

#endif
