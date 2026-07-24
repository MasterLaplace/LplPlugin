/**
 * @file EntityDelta.cpp
 * @brief Field-masked entity delta codec implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-24
 * @copyright MIT License
 */

#include <lpl/net/protocol/EntityDelta.hpp>

namespace lpl::net::protocol {

core::u8 computeFieldMask(const EntitySnapshot &prev, const EntitySnapshot &cur) noexcept
{
    core::u8 mask = 0;
    if (cur.px != prev.px)
        mask |= FieldPosX;
    if (cur.py != prev.py)
        mask |= FieldPosY;
    if (cur.pz != prev.pz)
        mask |= FieldPosZ;
    if (cur.sx != prev.sx)
        mask |= FieldSizeX;
    if (cur.sy != prev.sy)
        mask |= FieldSizeY;
    if (cur.sz != prev.sz)
        mask |= FieldSizeZ;
    if (cur.hp != prev.hp)
        mask |= FieldHp;
    return mask;
}

void writeEntityDelta(Bitstream &stream, const EntitySnapshot &cur, core::u8 mask)
{
    stream.writeU32(cur.id);
    stream.writeU8(mask);

    if (mask & FieldPosX)
        stream.writeFloat(cur.px);
    if (mask & FieldPosY)
        stream.writeFloat(cur.py);
    if (mask & FieldPosZ)
        stream.writeFloat(cur.pz);
    if (mask & FieldSizeX)
        stream.writeFloat(cur.sx);
    if (mask & FieldSizeY)
        stream.writeFloat(cur.sy);
    if (mask & FieldSizeZ)
        stream.writeFloat(cur.sz);
    if (mask & FieldHp)
        stream.writeI32(cur.hp);
}

core::Expected<void> readEntityDelta(Bitstream &stream, EntitySnapshot &inOut, core::u32 &outId, core::u8 &outMask)
{
    auto rId = stream.readU32();
    if (!rId.has_value())
        return core::makeError(rId.error().code(), rId.error().message());
    auto rMask = stream.readU8();
    if (!rMask.has_value())
        return core::makeError(rMask.error().code(), rMask.error().message());

    outId = rId.value();
    outMask = rMask.value();
    inOut.id = outId;

    // Each present field overwrites the corresponding component of the baseline;
    // absent ones keep whatever the receiver already believed.
    const auto readF = [&](float &dst) -> core::Expected<void> {
        auto r = stream.readFloat();
        if (!r.has_value())
            return core::makeError(r.error().code(), r.error().message());
        dst = r.value();
        return {};
    };

    if (outMask & FieldPosX)
        if (auto r = readF(inOut.px); !r.has_value())
            return r;
    if (outMask & FieldPosY)
        if (auto r = readF(inOut.py); !r.has_value())
            return r;
    if (outMask & FieldPosZ)
        if (auto r = readF(inOut.pz); !r.has_value())
            return r;
    if (outMask & FieldSizeX)
        if (auto r = readF(inOut.sx); !r.has_value())
            return r;
    if (outMask & FieldSizeY)
        if (auto r = readF(inOut.sy); !r.has_value())
            return r;
    if (outMask & FieldSizeZ)
        if (auto r = readF(inOut.sz); !r.has_value())
            return r;
    if (outMask & FieldHp)
    {
        auto r = stream.readI32();
        if (!r.has_value())
            return core::makeError(r.error().code(), r.error().message());
        inOut.hp = r.value();
    }

    return {};
}

} // namespace lpl::net::protocol
