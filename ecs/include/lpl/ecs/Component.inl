#ifndef LPL_ECS_COMPONENT_INL
#define LPL_ECS_COMPONENT_INL

namespace lpl::ecs {

constexpr ComponentLayout defaultLayout(ComponentId id) noexcept
{
    switch (id)
    {
    case ComponentId::Position: return {id, 12, 4};
    case ComponentId::Velocity: return {id, 12, 4};
    case ComponentId::Rotation: return {id, 16, 4};
    case ComponentId::AngularVelocity: return {id, 12, 4};
    case ComponentId::Mass: return {id, 4, 4};
    case ComponentId::AABB: return {id, 12, 4};
    case ComponentId::Health: return {id, 4, 4};
    case ComponentId::NetworkSync: return {id, 4, 4};
    case ComponentId::InputSnapshot: return {id, 4, 4};
    case ComponentId::PlayerTag: return {id, 1, 1};
    case ComponentId::SleepState: return {id, 1, 1};
    case ComponentId::BciInput: return {id, 12, 4};
    case ComponentId::Genome: return {id, 20, 4};
    case ComponentId::Creature: return {id, 8, 4};
    case ComponentId::Heading: return {id, 8, 4};
    case ComponentId::Historical: return {id, 12, 4};
    case ComponentId::WorldCell: return {id, 8, 4};
    default: return {id, 4, 4};
    }
}

} // namespace lpl::ecs

#endif // LPL_ECS_COMPONENT_INL
