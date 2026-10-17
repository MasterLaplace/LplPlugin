#ifndef LPL_ECS_COMPONENTREFLECTION_INL
#define LPL_ECS_COMPONENTREFLECTION_INL

namespace lpl::ecs {

constexpr core::u32 fieldSize(FieldType t) noexcept
{
    switch (t)
    {
    case FieldType::F32:
    case FieldType::I32:
    case FieldType::U32:
    case FieldType::Fixed32: return 4;
    case FieldType::U16: return 2;
    case FieldType::U8: return 1;
    case FieldType::Vec3F:
    case FieldType::Vec3Fixed: return 12;
    case FieldType::QuatF: return 16;
    }
    return 0;
}

constexpr core::u32 fieldAlign(FieldType t) noexcept
{
    switch (t)
    {
    case FieldType::U8: return 1;
    case FieldType::U16: return 2;
    default: return 4;
    }
}

constexpr DerivedLayout computeLayout(const ComponentSchema &schema) noexcept
{
    core::u32 size = 0;
    core::u32 align = 1;
    for (const FieldDesc &f : schema.fields)
    {
        const core::u32 end = f.offset + fieldSize(f.type);
        if (end > size)
            size = end;
        const core::u32 a = fieldAlign(f.type);
        if (a > align)
            align = a;
    }
    if (align != 0 && (size % align) != 0)
        size += align - (size % align);
    return {size, align};
}

namespace detail {

constexpr core::u32 fieldWidth(FieldType type) noexcept
{
    switch (type)
    {
    case FieldType::U8: return 1u;
    case FieldType::U16: return 2u;
    case FieldType::F32:
    case FieldType::I32:
    case FieldType::U32:
    case FieldType::Fixed32: return 4u;
    case FieldType::Vec3F:
    case FieldType::Vec3Fixed: return 12u;
    case FieldType::QuatF: return 16u;
    }
    return 4u;
}

constexpr bool schemaTilesComponent(const ComponentSchema &schema) noexcept
{
    core::u32 cursor = 0u;
    for (const FieldDesc &field : schema.fields)
    {
        if (field.offset != cursor)
            return false;
        cursor += fieldWidth(field.type);
    }
    return cursor == defaultLayout(schema.id).size;
}

constexpr bool everySchemaTilesItsComponent() noexcept
{
    for (const ComponentSchema &schema : kSchemas)
    {
        if (schema.fields.empty())
            continue;
        if (!schemaTilesComponent(schema))
            return false;
    }
    return true;
}

static_assert(everySchemaTilesItsComponent(),
              "a component schema does not tile its component: a field is missing, an offset is "
              "wrong, or the layout changed without the reflection following");

} // namespace detail

constexpr ComponentId componentIdByName(std::string_view name) noexcept
{
    for (const ComponentSchema &s : detail::kSchemas)
        if (s.name == name)
            return s.id;
    return ComponentId::Count;
}

} // namespace lpl::ecs

#endif // LPL_ECS_COMPONENTREFLECTION_INL
