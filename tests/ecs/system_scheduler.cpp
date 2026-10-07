#include <lpl/concurrency/IJobSystem.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/System.hpp>
#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/std/memory.hpp>
#include <lpl/testing/Test.hpp>

#include <span>

LPL_TEST_SUITE(system_scheduler);

namespace {

/**
 * @brief What the systems leave behind as they run, fixed by the order they run in.
 */
struct Trace {
    lpl::core::u32 mask = 0u;           /**< Bit `marker` set when that system ran. */
    lpl::core::u32 order[8] = {};       /**< Markers in the order the systems ran. */
    lpl::core::u32 ran = 0u;            /**< Systems that ran. */
    lpl::core::u32 phaseCallbacks = 0u; /**< Times the callback after the input phase fired. */
};

/**
 * @brief A system that records its marker in the trace when it runs.
 */
class MarkerSystem final : public lpl::ecs::ISystem {
public:
    MarkerSystem(Trace *trace, lpl::core::u32 marker, lpl::ecs::SchedulePhase phase,
                 std::span<const lpl::ecs::ComponentAccess> accesses)
        : _trace{trace}, _marker{marker}, _descriptor{"", phase, accesses}
    {
    }

    const lpl::ecs::SystemDescriptor &descriptor() const noexcept override { return _descriptor; }

    void execute([[maybe_unused]] lpl::core::f32 dt) override
    {
        _trace->mask |= (1u << _marker);
        _trace->order[_trace->ran++] = _marker;
    }

private:
    Trace *_trace;
    lpl::core::u32 _marker;
    lpl::ecs::SystemDescriptor _descriptor;
};

constexpr lpl::ecs::ComponentAccess kWritesPosition[] = {
    {lpl::ecs::ComponentId::Position, lpl::ecs::AccessMode::ReadWrite}
};
constexpr lpl::ecs::ComponentAccess kReadsPositionWritesVelocity[] = {
    {lpl::ecs::ComponentId::Position, lpl::ecs::AccessMode::ReadOnly },
    {lpl::ecs::ComponentId::Velocity, lpl::ecs::AccessMode::ReadWrite}
};
constexpr lpl::ecs::ComponentAccess kReadsVelocity[] = {
    {lpl::ecs::ComponentId::Velocity, lpl::ecs::AccessMode::ReadOnly}
};
constexpr lpl::ecs::ComponentAccess kWritesMass[] = {
    {lpl::ecs::ComponentId::Mass, lpl::ecs::AccessMode::ReadWrite}
};

} // namespace

/**
 * @brief Gate P1 scheduler: four systems over two phases, whose accesses force the order
 *        1, then 2 and 4, then 3, on the job system that runs inline.
 */
LPL_TEST(accesses_order_the_systems)
{
    Trace trace;
    lpl::concurrency::InlineJobSystem jobSystem;
    lpl::ecs::SystemScheduler scheduler{jobSystem};

    (void) scheduler.registerSystem(
        lpl::pmr::make_unique<MarkerSystem>(&trace, 1u, lpl::ecs::SchedulePhase::Input, kWritesPosition));
    (void) scheduler.registerSystem(lpl::pmr::make_unique<MarkerSystem>(&trace, 2u, lpl::ecs::SchedulePhase::Physics,
                                                                        kReadsPositionWritesVelocity));
    (void) scheduler.registerSystem(
        lpl::pmr::make_unique<MarkerSystem>(&trace, 3u, lpl::ecs::SchedulePhase::Physics, kReadsVelocity));
    (void) scheduler.registerSystem(
        lpl::pmr::make_unique<MarkerSystem>(&trace, 4u, lpl::ecs::SchedulePhase::Physics, kWritesMass));
    scheduler.setPhaseCallback(lpl::ecs::SchedulePhase::Input, [&trace]() { ++trace.phaseCallbacks; });

    test.check(scheduler.systemCount() == 4u, "four systems are registered");
    if (!test.check(scheduler.buildGraph().has_value(), "their accesses order without a cycle"))
        return;

    scheduler.tick(1.0f);

    test.check(trace.ran == 4u && trace.mask == 0x1Eu, "each system runs once");
    test.check(trace.order[0] == 1u, "the input system runs first");
    test.check(trace.order[3] == 3u, "and the one reading the velocity the second writes runs last");
    test.check(trace.phaseCallbacks == 1u, "the callback after the input phase fires once");
}
