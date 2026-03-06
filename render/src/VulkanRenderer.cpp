#include <lpl/core/Log.hpp>
#include <lpl/render/Mesh.hpp>
#include <lpl/render/VulkanRenderer.hpp>

#include "vk/wrapper/Wrapper.hpp"
#include "wrapper/Wrapper.hpp"

namespace lpl::render::vk {

// ---------------------------------------------------------------------------
// Internal helper: build a unit cube centred at the origin.
//
// 24 unique vertices (4 per face, flat normals), 36 indices (12 triangles).
// ---------------------------------------------------------------------------
static ::lpl::render::Mesh makeCube()
{
    using V = ::lpl::render::Vertex;
    using u3 = ::lpl::core::u32;

    // clang-format off
    static const V kVerts[24] = {
        // Front   (Z+)
        {{-0.5f,-0.5f, 0.5f},{0,0,1},0.0f,0.0f},
        {{ 0.5f,-0.5f, 0.5f},{0,0,1},1.0f,0.0f},
        {{ 0.5f, 0.5f, 0.5f},{0,0,1},1.0f,1.0f},
        {{-0.5f, 0.5f, 0.5f},{0,0,1},0.0f,1.0f},
        // Back    (Z-)
        {{ 0.5f,-0.5f,-0.5f},{0,0,-1},0.0f,0.0f},
        {{-0.5f,-0.5f,-0.5f},{0,0,-1},1.0f,0.0f},
        {{-0.5f, 0.5f,-0.5f},{0,0,-1},1.0f,1.0f},
        {{ 0.5f, 0.5f,-0.5f},{0,0,-1},0.0f,1.0f},
        // Left    (X-)
        {{-0.5f,-0.5f,-0.5f},{-1,0,0},0.0f,0.0f},
        {{-0.5f,-0.5f, 0.5f},{-1,0,0},1.0f,0.0f},
        {{-0.5f, 0.5f, 0.5f},{-1,0,0},1.0f,1.0f},
        {{-0.5f, 0.5f,-0.5f},{-1,0,0},0.0f,1.0f},
        // Right   (X+)
        {{ 0.5f,-0.5f, 0.5f},{1,0,0},0.0f,0.0f},
        {{ 0.5f,-0.5f,-0.5f},{1,0,0},1.0f,0.0f},
        {{ 0.5f, 0.5f,-0.5f},{1,0,0},1.0f,1.0f},
        {{ 0.5f, 0.5f, 0.5f},{1,0,0},0.0f,1.0f},
        // Top     (Y+)
        {{-0.5f, 0.5f, 0.5f},{0,1,0},0.0f,0.0f},
        {{ 0.5f, 0.5f, 0.5f},{0,1,0},1.0f,0.0f},
        {{ 0.5f, 0.5f,-0.5f},{0,1,0},1.0f,1.0f},
        {{-0.5f, 0.5f,-0.5f},{0,1,0},0.0f,1.0f},
        // Bottom  (Y-)
        {{-0.5f,-0.5f,-0.5f},{0,-1,0},0.0f,0.0f},
        {{ 0.5f,-0.5f,-0.5f},{0,-1,0},1.0f,0.0f},
        {{ 0.5f,-0.5f, 0.5f},{0,-1,0},1.0f,1.0f},
        {{-0.5f,-0.5f, 0.5f},{0,-1,0},0.0f,1.0f},
    };
    // clang-format on

    std::vector<::lpl::render::Vertex> verts(std::begin(kVerts), std::end(kVerts));

    std::vector<u3> idx;
    idx.reserve(36);
    for (u3 face = 0; face < 6; ++face)
    {
        u3 b = face * 4;
        idx.insert(idx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }

    ::lpl::render::Mesh mesh;
    mesh.setVertices(std::move(verts));
    mesh.setIndices(std::move(idx));
    return mesh;
}

VulkanRenderer::VulkanRenderer() = default;

VulkanRenderer::~VulkanRenderer() { shutdown(); }

core::Expected<void> VulkanRenderer::init(core::u32 /*width*/, core::u32 /*height*/)
{
    _wrapper = std::make_unique<Wrapper>();
    // Note: The original engine created the instance immediately here.
    // In reality, apps/client/main.cpp must provide the window.
    // We will configure the pipeline lazily when textures and models are ready.
    core::Log::info("VulkanRenderer initialized.");
    return core::Expected<void>{};
}

void VulkanRenderer::beginFrame()
{
    // Usually clears or waits for semaphores
}

void VulkanRenderer::endFrame()
{
    auto result = _wrapper->DrawFrame();
    if (result == Wrapper::Result::NeedResize)
    {
        // Window pointer is held internally by Wrapper's instance
    }
}

void VulkanRenderer::resize(core::u32 /*width*/, core::u32 /*height*/)
{
    // The Wrapper already intercepts resize via GLFW callback,
    // but we might need explicit bounds setting here.
}

void VulkanRenderer::initVulkanContext(GLFWwindow *window)
{
    _wrapper->CreateInstance(window, "LplPlugin Client", 800, 600);

    // ── Shaders ──────────────────────────────────────────────────────────────
    // LPL_SHADER_DIR is defined by render/xmake.lua as an absolute path so the
    // binary works regardless of the current working directory.
    const std::string shaderDir =
#ifdef LPL_SHADER_DIR
        std::string{LPL_SHADER_DIR};
#else
        std::string{"shaders/"}; // fallback: CWD-relative
#endif
    _wrapper->AddShader(shaderDir + "vert.spv", "main", Wrapper::ShaderType::VERTEX);
    _wrapper->AddShader(shaderDir + "frag.spv", "main", Wrapper::ShaderType::FRAGMENT);

    // ── Texture ──────────────────────────────────────────────────────────────
    const std::string assetDir =
#ifdef LPL_ASSET_DIR
        std::string{LPL_ASSET_DIR};
#else
        std::string{"assets/"};
#endif
    core::u32 texId{};
    _wrapper->AddTexture(assetDir + "textures/white.png", texId);

    // ── Debug cube mesh ───────────────────────────────────────────────────────
    core::u32 cubeId{};
    auto cubeMesh = makeCube();
    _wrapper->AddModel(cubeMesh, "debug_cube", cubeId);
    _wrapper->BindTexture(texId, cubeId);

    _wrapper->EnableDepthTest(true);
    _wrapper->CreatePipeline();
}

void VulkanRenderer::shutdown()
{
    // Guard against double-destroy: Engine::shutdown() calls shutdown() then
    // reset() on the unique_ptr, which would invoke the destructor (and thus
    // shutdown() again). Nullifying the wrapper makes the call idempotent.
    if (!_wrapper)
        return;
    _wrapper->Destroy();
    _wrapper.reset();
}

Wrapper &VulkanRenderer::getWrapper() noexcept { return *_wrapper; }

} // namespace lpl::render::vk
