#ifndef LPL_ENGINE_GAMEPROFILE_INL
#define LPL_ENGINE_GAMEPROFILE_INL

namespace lpl::engine {

inline Config::Builder &applyGameProfile(Config::Builder &builder, GameProfile profile)
{
    switch (profile)
    {
    case GameProfile::Mmorpg:
        builder.enablePhysics(true)
            .enableNetworking(true)
            .interestRadius(math::Fixed32::fromFloat(80.0f))
            .lodNearRadius(math::Fixed32::fromFloat(30.0f))
            .lodFarInterval(4)
            .worldExtent(math::Fixed32::fromFloat(1000.0f))
            .lodFarPosBits(16)
            .keyframeInterval(60)
            .bandwidthBudgetBytes(48 * 1024)
            .reliableBaseline(false);
        break;

    case GameProfile::Fps:
        builder.enablePhysics(true)
            .enableNetworking(true)
            .interestRadius(math::Fixed32::fromFloat(150.0f))
            .lodNearRadius(math::Fixed32::zero())
            .worldExtent(math::Fixed32::zero())
            .keyframeInterval(1000)
            .bandwidthBudgetBytes(0)
            .reliableBaseline(true);
        break;

    case GameProfile::Rts:
        builder.enablePhysics(true)
            .enableNetworking(true)
            .interestRadius(math::Fixed32::fromFloat(400.0f))
            .lodNearRadius(math::Fixed32::fromFloat(120.0f))
            .lodFarInterval(6)
            .worldExtent(math::Fixed32::fromFloat(2000.0f))
            .lodFarPosBits(16)
            .keyframeInterval(90)
            .bandwidthBudgetBytes(96 * 1024)
            .reliableBaseline(false);
        break;

    case GameProfile::Fighting:
        builder.enablePhysics(true)
            .enableNetworking(true)
            .interestRadius(math::Fixed32::zero())
            .lodNearRadius(math::Fixed32::zero())
            .worldExtent(math::Fixed32::zero())
            .keyframeInterval(1000)
            .reliableBaseline(false);
        break;

    case GameProfile::CoOp:
        builder.enablePhysics(true)
            .enableNetworking(true)
            .interestRadius(math::Fixed32::zero())
            .lodNearRadius(math::Fixed32::zero())
            .worldExtent(math::Fixed32::zero())
            .keyframeInterval(60)
            .reliableBaseline(false);
        break;
    }
    return builder;
}

inline bool parseGameProfile(const char *name, GameProfile &outProfile)
{
    if (name == nullptr)
        return false;
    struct Entry {
        const char *name;
        GameProfile profile;
    };
    static const Entry kTable[] = {
        {"mmorpg",   GameProfile::Mmorpg  },
        {"fps",      GameProfile::Fps     },
        {"rts",      GameProfile::Rts     },
        {"fighting", GameProfile::Fighting},
        {"coop",     GameProfile::CoOp    },
    };
    for (const auto &e : kTable)
        if (std::strcmp(name, e.name) == 0)
        {
            outProfile = e.profile;
            return true;
        }
    return false;
}

inline const char *gameProfileName(GameProfile profile) noexcept
{
    switch (profile)
    {
    case GameProfile::Mmorpg: return "MMORPG";
    case GameProfile::Fps: return "FPS";
    case GameProfile::Rts: return "RTS";
    case GameProfile::Fighting: return "Fighting";
    case GameProfile::CoOp: return "Co-op";
    }
    return "unknown";
}

} // namespace lpl::engine

#endif // LPL_ENGINE_GAMEPROFILE_INL
