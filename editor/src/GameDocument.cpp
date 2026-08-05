/**
 * @file GameDocument.cpp
 * @brief Implementation of the four-stage game document.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/editor/GameDocument.hpp>

#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/editor/Json.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

namespace lpl::editor {

namespace {

constexpr const char *kFormat = "lplscene/1";

struct NamedKind {
    const char *name;
    ResourceKind kind;
};

constexpr NamedKind kKinds[] = {
    {"other",   ResourceKind::Other  },
    {"texture", ResourceKind::Texture},
    {"font",    ResourceKind::Font   },
    {"sound",   ResourceKind::Sound  },
    {"music",   ResourceKind::Music  },
    {"shader",  ResourceKind::Shader },
    {"data",    ResourceKind::Data   },
};

struct NamedScope {
    const char *name;
    ResourceScope scope;
};

constexpr NamedScope kScopes[] = {
    {"shared", ResourceScope::Shared},
    {"client", ResourceScope::Client},
    {"server", ResourceScope::Server},
};

ResourceKind kindFromName(std::string_view name)
{
    for (const NamedKind &entry : kKinds)
        if (name == entry.name)
            return entry.kind;
    return ResourceKind::Other;
}

ResourceScope scopeFromName(std::string_view name)
{
    for (const NamedScope &entry : kScopes)
        if (name == entry.name)
            return entry.scope;
    return ResourceScope::Shared;
}

/// Reads a string member, or @p fallback when absent.
std::string stringOr(const detail::JVal &object, const char *key, std::string fallback = {})
{
    const detail::JVal *value = object.find(key);
    return (value != nullptr && value->t == detail::JVal::T::Str) ? value->str : std::move(fallback);
}

/// Reads a boolean member, or @p fallback when absent.
bool boolOr(const detail::JVal &object, const char *key, bool fallback)
{
    const detail::JVal *value = object.find(key);
    return (value != nullptr && value->t == detail::JVal::T::Bool) ? value->b : fallback;
}

/// Re-emits a member as JSON text, or @p fallback when absent.
std::string subDocumentOr(const detail::JVal &object, const char *key, const char *fallback)
{
    const detail::JVal *value = object.find(key);
    return (value != nullptr) ? detail::emit(*value) : std::string{fallback};
}

void appendEscaped(std::string &out, std::string_view text)
{
    for (const char c : text)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
}

void appendString(std::string &out, const char *key, std::string_view value)
{
    out += "\"";
    out += key;
    out += "\":\"";
    appendEscaped(out, value);
    out += "\"";
}

/// Reads the resource table of a scene. Accepts both the flat form
/// (`"resources":[{name,path,kind,scope}]`) and Flakkari's per-kind arrays
/// (`"textures":[...]`), because that is the shape existing config.cfg files
/// have and refusing them would make migration a rewrite.
void parseResources(const detail::JVal &sceneObject, SceneDescription &scene)
{
    if (const detail::JVal *flat = sceneObject.find("resources"); flat != nullptr && flat->t == detail::JVal::T::Arr)
    {
        for (const detail::JVal &entry : flat->arr)
        {
            if (entry.t != detail::JVal::T::Obj)
                continue;
            ResourceEntry resource;
            resource.name = stringOr(entry, "name");
            resource.path = stringOr(entry, "path");
            resource.kind = kindFromName(stringOr(entry, "kind", "other"));
            resource.scope = scopeFromName(stringOr(entry, "scope", "shared"));
            if (!resource.name.empty())
                scene.resources.push_back(std::move(resource));
        }
    }

    // Flakkari's per-kind arrays. A texture, a font, a sound and a music track
    // are presentation by definition, so their scope is implied rather than
    // spelled out — a server reading such a document skips them all without the
    // author having had to say so.
    struct KindArray {
        const char *key;
        ResourceKind kind;
        ResourceScope scope;
    };
    constexpr KindArray kArrays[] = {
        {"textures", ResourceKind::Texture, ResourceScope::Client},
        {"fonts",    ResourceKind::Font,    ResourceScope::Client},
        {"sounds",   ResourceKind::Sound,   ResourceScope::Client},
        {"music",    ResourceKind::Music,   ResourceScope::Client},
        {"shaders",  ResourceKind::Shader,  ResourceScope::Client},
    };
    for (const KindArray &array : kArrays)
    {
        const detail::JVal *node = sceneObject.find(array.key);
        if (node == nullptr || node->t != detail::JVal::T::Obj)
            continue;
        // Flakkari spells these as name -> path maps.
        for (const auto &binding : node->obj)
        {
            if (binding.second.t != detail::JVal::T::Str)
                continue;
            scene.resources.push_back(ResourceEntry{binding.first, binding.second.str, array.kind, array.scope});
        }
    }
}

/// Reads one scene object (or a whole document treated as an implicit scene).
SceneDescription parseScene(const detail::JVal &object, std::string defaultName)
{
    SceneDescription scene;
    scene.name = stringOr(object, "name", std::move(defaultName));

    if (const detail::JVal *systems = object.find("systems"); systems != nullptr && systems->t == detail::JVal::T::Arr)
        for (const detail::JVal &entry : systems->arr)
            if (entry.t == detail::JVal::T::Str)
                scene.systems.push_back(entry.str); // order preserved: it IS the schedule

    parseResources(object, scene);

    if (const detail::JVal *procedural = object.find("procedural");
        procedural != nullptr && procedural->t == detail::JVal::T::Obj)
    {
        // Reuse the recipe reader rather than duplicating twenty field names:
        // a second parser would drift from the baker's the first time a knob is
        // added, and the drift would silently change generated worlds.
        if (parseProceduralBlock(detail::emit(*procedural), scene.recipe))
            scene.hasRecipe = true;
    }

    // The ecosystem, read straight off the scene object: unlike the recipe it
    // needs no wrapping, because parseSceneLiving takes the scene rather than a
    // whole document.
    if (parseSceneLiving(object, scene.living))
        scene.hasLiving = true;
    if (parseSceneView(object, scene.view))
        scene.hasView = true;

    scene.templatesJson = subDocumentOr(object, "templates", "{}");
    scene.entitiesJson = subDocumentOr(object, "entities", "[]");
    return scene;
}

} // namespace

std::string_view resourceKindName(ResourceKind kind) noexcept
{
    for (const NamedKind &entry : kKinds)
        if (entry.kind == kind)
            return entry.name;
    return "other";
}

std::string_view resourceScopeName(ResourceScope scope) noexcept
{
    for (const NamedScope &entry : kScopes)
        if (entry.scope == scope)
            return entry.name;
    return "shared";
}

const SceneDescription *GameDocument::findScene(std::string_view name) const
{
    for (const SceneDescription &scene : scenes)
        if (scene.name == name)
            return &scene;
    return nullptr;
}

const SceneDescription *GameDocument::startScene() const
{
    if (!metadata.startScene.empty())
        if (const SceneDescription *named = findScene(metadata.startScene); named != nullptr)
            return named;
    return scenes.empty() ? nullptr : &scenes.front();
}

core::Expected<GameDocument> parseGameDocument(std::string_view text)
{
    bool ok = false;
    const detail::JVal root = detail::parse(text, &ok);
    if (!ok || root.t != detail::JVal::T::Obj)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"malformed .lplscene root"});

    const detail::JVal *format = root.find("format");
    if (format == nullptr || format->t != detail::JVal::T::Str || format->str != kFormat)
        return core::makeError(core::ErrorCode::kNotSupported, lpl::pmr::string{"unsupported .lplscene format"});

    GameDocument document;

    if (const detail::JVal *meta = root.find("metadata"); meta != nullptr && meta->t == detail::JVal::T::Obj)
    {
        document.metadata.title = stringOr(*meta, "title");
        document.metadata.version = stringOr(*meta, "version");
        document.metadata.profile = stringOr(*meta, "profile");
        document.metadata.startScene = stringOr(*meta, "startScene");
        document.metadata.online = boolOr(*meta, "online", false);
        document.metadata.minPlayers = static_cast<core::u32>(meta->numOr("minPlayers", 1.0));
        document.metadata.maxPlayers = static_cast<core::u32>(meta->numOr("maxPlayers", 1.0));
        document.metadata.maxInstances = static_cast<core::u32>(meta->numOr("maxInstances", 1.0));
    }

    if (const detail::JVal *scenes = root.find("scenes"); scenes != nullptr && scenes->t == detail::JVal::T::Arr)
    {
        core::u32 index = 0u;
        for (const detail::JVal &entry : scenes->arr)
        {
            if (entry.t != detail::JVal::T::Obj)
                continue;
            document.scenes.push_back(parseScene(entry, "scene" + std::to_string(index)));
            ++index;
        }
    }
    else
    {
        // No "scenes" array: the root IS the scene. Every document written
        // before this stage existed has that shape, and they must keep working.
        document.scenes.push_back(parseScene(root, "main"));
    }

    if (document.scenes.empty())
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"document declares no scene"});

    return document;
}

std::string emitGameDocument(const GameDocument &document)
{
    std::string out = "{\"format\":\"";
    out += kFormat;
    out += "\",\"metadata\":{";
    appendString(out, "title", document.metadata.title);
    out += ",";
    appendString(out, "version", document.metadata.version);
    out += ",";
    appendString(out, "profile", document.metadata.profile);
    out += ",";
    appendString(out, "startScene", document.metadata.startScene);
    out += ",\"online\":";
    out += document.metadata.online ? "true" : "false";
    out += ",\"minPlayers\":" + std::to_string(document.metadata.minPlayers);
    out += ",\"maxPlayers\":" + std::to_string(document.metadata.maxPlayers);
    out += ",\"maxInstances\":" + std::to_string(document.metadata.maxInstances);
    out += "},\"scenes\":[";

    for (core::usize s = 0u; s < document.scenes.size(); ++s)
    {
        const SceneDescription &scene = document.scenes[s];
        if (s != 0u)
            out += ",";
        out += "{";
        appendString(out, "name", scene.name);

        out += ",\"systems\":[";
        for (core::usize i = 0u; i < scene.systems.size(); ++i)
        {
            if (i != 0u)
                out += ",";
            out += "\"";
            appendEscaped(out, scene.systems[i]);
            out += "\"";
        }
        out += "]";

        // Always emitted in the flat form, whatever shape it was read in: one
        // way to write it means one way to read it back.
        out += ",\"resources\":[";
        for (core::usize i = 0u; i < scene.resources.size(); ++i)
        {
            const ResourceEntry &resource = scene.resources[i];
            if (i != 0u)
                out += ",";
            out += "{";
            appendString(out, "name", resource.name);
            out += ",";
            appendString(out, "path", resource.path);
            out += ",";
            appendString(out, "kind", resourceKindName(resource.kind));
            out += ",";
            appendString(out, "scope", resourceScopeName(resource.scope));
            out += "}";
        }
        out += "]";

        if (scene.hasRecipe)
            out += ",\"procedural\":" + emitSceneRecipe(scene.recipe);
        if (scene.hasLiving)
            out += ",\"living\":" + emitSceneLiving(scene.living);
        if (scene.hasView)
            out += ",\"view\":" + emitSceneView(scene.view);

        out += ",\"templates\":" + scene.templatesJson;
        out += ",\"entities\":" + scene.entitiesJson;
        out += "}";
    }
    out += "]}";
    return out;
}

core::Expected<core::u32> instantiateScene(const SceneDescription &scene, ecs::Registry &registry)
{
    core::u32 created = 0u;

    // Procedural first, explicit entities on top — see the header's note on why
    // this order is a contract and not an implementation detail.
    if (scene.hasRecipe)
        created += procgen::bakeWorld(registry, scene.recipe).entityCount;

    const std::string sub = std::string{"{\"format\":\""} + kFormat + "\",\"templates\":" + scene.templatesJson +
                            ",\"entities\":" + scene.entitiesJson + "}";
    const auto loaded = fromLplScene(sub, registry);
    if (!loaded)
        return std::unexpected(loaded.error());

    return created + loaded.value();
}

std::vector<const ResourceEntry *> resourcesFor(const SceneDescription &scene, ResourceScope consumer)
{
    std::vector<const ResourceEntry *> selected;
    for (const ResourceEntry &resource : scene.resources)
        if (resource.scope == ResourceScope::Shared || resource.scope == consumer)
            selected.push_back(&resource);
    return selected;
}

} // namespace lpl::editor
