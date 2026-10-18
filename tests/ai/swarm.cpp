#include <lpl/ai/AntColony.hpp>
#include <lpl/ai/StigmergyField.hpp>
#include <lpl/ai/Swarm.hpp>
#include <lpl/procgen/Grid.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(swarm);

namespace {

/**
 * @brief Moves an ant asks for, sorted by where they lead.
 */
struct MoveTally {
    lpl::core::u32 explored{0u}; /**< Moves where the ant ignored the field. */
    lpl::core::u32 stayed{0u};   /**< Moves that returned no direction. */
    lpl::core::u32 open{0u};     /**< Moves into a cell inside the field that no wall blocks. */
    lpl::core::u32 walled{0u};   /**< Moves into a wall. */
    lpl::core::u32 outside{0u};  /**< Moves off the field. */
};

/**
 * @brief Asks an ant that always explores for its move, many times, from one cell.
 *
 * @param field The field, walls included.
 * @param walls The same walls, to sort the moves by.
 * @param x     The ant's column.
 * @param z     The ant's row.
 * @param moves How many moves to ask for.
 * @return Where the moves led.
 */
[[nodiscard]] MoveTally tallyExploringMoves(const lpl::ai::StigmergyField &field,
                                            const lpl::procgen::Grid<lpl::core::u8> &walls, lpl::core::u32 x,
                                            lpl::core::u32 z, lpl::core::u32 moves)
{
    lpl::ai::AntParams params;
    params.explore16 = 16u;
    lpl::core::u32 stream = 0x1234u;
    MoveTally tally;
    for (lpl::core::u32 i = 0u; i < moves; ++i)
    {
        bool explored = false;
        const lpl::core::u32 direction = lpl::ai::chooseAntMove(field, params, x, z, stream, explored);
        tally.explored += explored ? 1u : 0u;
        if (direction == lpl::ai::StigmergyField::kNoDirection)
        {
            ++tally.stayed;
            continue;
        }
        const lpl::core::i32 nx = static_cast<lpl::core::i32>(x) + lpl::procgen::kNeighbor8X[direction];
        const lpl::core::i32 nz = static_cast<lpl::core::i32>(z) + lpl::procgen::kNeighbor8Z[direction];
        if (nx < 0 || nz < 0 || static_cast<lpl::core::u32>(nx) >= field.width() ||
            static_cast<lpl::core::u32>(nz) >= field.depth())
            ++tally.outside;
        else if (walls.at(static_cast<lpl::core::u32>(nx), static_cast<lpl::core::u32>(nz)) != 0u)
            ++tally.walled;
        else
            ++tally.open;
    }
    return tally;
}

} // namespace

/**
 * @brief An exploring ant with a wall on every side stays where it is.
 *
 * @details An explorer took `(stream >> 8) % 8` whatever the cell there held, so it stepped into a wall, where
 *          nothing it deposits is kept.
 */
LPL_TEST(an_exploring_ant_walled_in_stays_where_it_is)
{
    lpl::procgen::Grid<lpl::core::u8> walls{3u, 3u, lpl::core::u8{1}};
    walls.at(1u, 1u) = 0u;
    lpl::ai::StigmergyField field{3u, 3u, 1u};
    field.setObstacles(walls);

    const MoveTally tally = tallyExploringMoves(field, walls, 1u, 1u, 64u);

    test.check(tally.explored == 64u, "every move explored");
    test.check(tally.walled == 0u, "no move leads into a wall");
    test.check(tally.stayed == 64u, "so the ant stays where it is");
}

/**
 * @brief An exploring ant beside walls steps only into the open cells, and still steps.
 *
 * @details The walls stand on the four sides and the corners are open, so a ruled-out wall leaves somewhere to go:
 *          an explorer that never moves would also never step into a wall.
 */
LPL_TEST(an_exploring_ant_steps_only_into_open_cells)
{
    lpl::procgen::Grid<lpl::core::u8> walls{5u, 5u, lpl::core::u8{0}};
    walls.at(2u, 1u) = 1u;
    walls.at(1u, 2u) = 1u;
    walls.at(3u, 2u) = 1u;
    walls.at(2u, 3u) = 1u;
    lpl::ai::StigmergyField field{5u, 5u, 1u};
    field.setObstacles(walls);

    const MoveTally tally = tallyExploringMoves(field, walls, 2u, 2u, 256u);

    test.check(tally.walled == 0u, "no move leads into a wall");
    test.check(tally.open != 0u, "the ant still explores the open corners");
    test.measure("open_moves", tally.open);
}

/**
 * @brief An exploring ant in a corner of the field never steps off it.
 */
LPL_TEST(an_exploring_ant_stays_inside_the_field)
{
    const lpl::procgen::Grid<lpl::core::u8> walls{4u, 4u, lpl::core::u8{0}};
    lpl::ai::StigmergyField field{4u, 4u, 1u};
    field.setObstacles(walls);

    const MoveTally tally = tallyExploringMoves(field, walls, 0u, 0u, 256u);

    test.check(tally.outside == 0u, "no move leads off the field");
    test.check(tally.open != 0u, "the ant still explores inward");
}

/**
 * @brief A colony's foragers never stand inside a wall, on a field where a third of the cells are walls.
 *
 * @details AntColony::step moved an agent wherever chooseAntMove pointed, checking only the bounds, so an explorer
 *          walked into the walls the trail it follows goes around.
 */
LPL_TEST(a_colony_never_stands_inside_a_wall)
{
    constexpr lpl::core::u32 kSide = 16u;
    lpl::procgen::Grid<lpl::core::u8> walls{kSide, kSide, lpl::core::u8{0}};
    for (lpl::core::u32 z = 0u; z < kSide; ++z)
        for (lpl::core::u32 x = 0u; x < kSide; ++x)
            walls.at(x, z) = (x + z) % 3u == 0u ? 1u : 0u;
    lpl::ai::StigmergyField field{kSide, kSide, 1u};
    field.setObstacles(walls);

    lpl::ai::AntColonyParams params;
    params.agents = 24u;
    params.forageRange = 12u;
    params.ants.explore16 = 8u;
    lpl::ai::AntColony colony;
    colony.reset(field, kSide, kSide, params, 4u, 4u);

    lpl::core::u32 explored = 0u;
    lpl::core::u32 inWalls = 0u;
    lpl::core::u32 awayFromNest = 0u;
    for (lpl::core::u32 tick = 0u; tick < 200u; ++tick)
    {
        colony.step(field);
        explored += colony.explored();
        for (lpl::core::u32 i = 0u; i < colony.agentCount(); ++i)
        {
            inWalls += field.passable(colony.agentX(i), colony.agentZ(i)) ? 0u : 1u;
            awayFromNest += colony.agentX(i) != colony.nestX() || colony.agentZ(i) != colony.nestZ() ? 1u : 0u;
        }
    }

    test.check(explored != 0u, "the foragers explored");
    test.check(awayFromNest != 0u, "and left the nest");
    test.check(inWalls == 0u, "and none of them ever stood inside a wall");
    test.measure("agent_ticks_in_walls", inWalls);
}
