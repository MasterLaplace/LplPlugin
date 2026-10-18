#include <lpl/engine/Server.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/net/Endpoint.hpp>

#    include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

/**
 * @brief A wrong digest for an in-history tick is detected as a divergence.
 *
 * @details A client reports a digest for one of its past ticks that disagrees with what the server held for
 *          that tick. Scans back for the first in-history tick that a deliberately wrong digest diverges from
 *          (the empty world hashes to a fixed value each tick, so any different digest diverges).
 *
 * @param server      The server, ticked a few times.
 * @param world       The instance it hosts.
 * @param wrongDigest The deliberately wrong digest.
 * @param targetTick  Receives the divergent tick found.
 */
void checkWrongDigestIsDetectedAsDivergence(const lpl::engine::Server &server, lpl::engine::Server::WorldId world,
                                            lpl::core::u64 wrongDigest, lpl::core::u64 &targetTick)
{
    bool foundDivergent = false;
    for (lpl::core::u64 t = server.currentTick(); t-- > 0;)
    {
        if (server.checkClientHash(world, t, wrongDigest) == lpl::engine::Server::DesyncVerdict::Diverged)
        {
            targetTick = t;
            foundDivergent = true;
            break;
        }
    }
    check(foundDivergent, "a wrong digest for an in-history tick is detected as a divergence");
}

/**
 * @brief Nothing is captured until a report is actually processed.
 *
 * @details No divergence has been *captured* yet — detection via checkClientHash is pure; capture happens
 *          only when a report flows through the tick.
 *
 * @param server The server, before any report reached it.
 */
void checkNothingIsCapturedUntilAReportIsProcessed(const lpl::engine::Server &server)
{
    lpl::engine::Server::DesyncReport before{};
    check(!server.lastDesyncReport(before), "nothing is captured until a report is actually processed");
}

/**
 * @brief A processed report of a divergence is counted and captured for post-mortem.
 *
 * @details Pushes the report into the instance's queue, exactly where a decoded StateHashReport packet lands,
 *          then ticks so the server consumes it.
 *
 * @param server      The server.
 * @param world       The instance it hosts.
 * @param targetTick  The divergent tick the report names.
 * @param wrongDigest The digest the report carries.
 * @param dt          The tick step.
 */
void checkProcessedReportIsCountedAndCaptured(lpl::engine::Server &server, lpl::engine::Server::WorldId world,
                                              lpl::core::u64 targetTick, lpl::core::u64 wrongDigest, float dt)
{
    const auto reporter = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 51000);
    lpl::engine::StateHashReportEvent report{};
    report.source = reporter;
    report.tick = targetTick;
    report.digest = wrongDigest;
    server.queues(world)->stateHashReports.push(std::move(report));

    const lpl::core::u64 desyncsBefore = server.desyncCount();
    server.tick(dt);

    check(server.desyncCount() == desyncsBefore + 1, "the divergence is counted");

    lpl::engine::Server::DesyncReport captured{};
    check(server.lastDesyncReport(captured), "the divergence is captured for post-mortem");
    check(captured.tick == targetTick, "the captured report names the divergent tick");
    check(captured.clientDigest == wrongDigest, "it records the client's reported digest");
    check(captured.serverDigest != captured.clientDigest, "server and client digests differ — that IS the divergence");
    check(captured.instance == world, "it records which instance diverged");
    check(captured.source == reporter, "it records who reported it");
}

/**
 * @brief A matching report is NOT flagged.
 *
 * @details Feeds the server its own digest, that of the latest folded tick, and confirms it agrees for a
 *          tick of its history (no false positive).
 *
 * @param server The server.
 * @param world  The instance it hosts.
 */
void checkMatchingReportIsNotFlagged(const lpl::engine::Server &server, lpl::engine::Server::WorldId world)
{
    lpl::core::u64 matchTick = 0;
    bool foundMatch = false;
    const lpl::core::u64 serverNow = server.stateHash(world);
    for (lpl::core::u64 t = server.currentTick(); t-- > 0;)
    {
        if (server.checkClientHash(world, t, serverNow) == lpl::engine::Server::DesyncVerdict::Match)
        {
            matchTick = t;
            foundMatch = true;
            break;
        }
    }
    check(foundMatch, "the server agrees with a client that reports the correct digest (no false positive)");
    (void) matchTick;
}

} // namespace

/**
 * @brief Desync detection + diagnostic capture test (book §6.4).
 *
 * @details Proves the server catches a client whose simulation diverged from the authoritative one, and
 *          captures the divergence for post-mortem: the tick, both digests, and who reported it (§6.4.2). A
 *          client reports a digest for a past tick; the server looks that tick up in its history and, when the
 *          digests disagree, records a DesyncReport — the hook a diagnostic tool hangs off. The server keeps a
 *          replay snapshot each tick for post-mortem.
 */
int main()
{
    std::printf("== desync detection + diagnostics ==\n");

    auto config = lpl::engine::Config::Builder{}
                      .serverMode(true)
                      .tickRate(60)
                      .serverPort(45996)
                      .replaySnapshotInterval(1)
                      .build();
    lpl::engine::Server server{config};
    check(server.init().has_value(), "server opens its socket");

    const auto world = server.addWorld(lpl::pmr::make_unique<lpl::engine::World>());
    check(world != lpl::engine::Server::kInvalidWorldId, "an instance is hosted");

    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 6; ++i)
        server.tick(dt);

    const lpl::core::u64 wrongDigest = 0xDEADBEEFCAFEBABEull;
    lpl::core::u64 targetTick = 0;
    checkWrongDigestIsDetectedAsDivergence(server, world, wrongDigest, targetTick);
    checkNothingIsCapturedUntilAReportIsProcessed(server);
    checkProcessedReportIsCountedAndCaptured(server, world, targetTick, wrongDigest, dt);
    checkMatchingReportIsNotFlagged(server, world);

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
