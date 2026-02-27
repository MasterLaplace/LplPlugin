// --- LAPLACE CLIENT (ANDROID PORT) --- //
// File: visual_android.cpp
// Description: Client OpenGL ES 2.0 pour Android.
//              Utilise le Core engine (identique en architecture au client desktop).
//              Se connecte au serveur LplPlugin via UDP 7777 (WiFi LAN).
//              Touch input → InputManager → MSG_INPUTS.
// Auteur: MasterLaplace

#include <android/log.h>
#include <android_native_app_glue.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include "Core.hpp"
#include "Systems.hpp"

// ─── Logging ──────────────────────────────────────────────────

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO,  "LplVisual", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN,  "LplVisual", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "LplVisual", __VA_ARGS__))

// ─── Mat4 (minimal, column-major for GLES) ────────────────────

struct Mat4 {
    float m[16];

    static Mat4 identity()
    {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static Mat4 perspective(float fovRad, float aspect, float near, float far)
    {
        Mat4 r{};
        float t = tanf(fovRad / 2.0f);
        r.m[0]  = 1.0f / (aspect * t);
        r.m[5]  = 1.0f / t;
        r.m[10] = -(far + near) / (far - near);
        r.m[11] = -1.0f;
        r.m[14] = -(2.0f * far * near) / (far - near);
        return r;
    }

    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 upDir)
    {
        Vec3 f = (center - eye).normalize();
        Vec3 s = f.cross(upDir).normalize();
        Vec3 u = s.cross(f);
        Mat4 r = identity();
        r.m[0]  =  s.x; r.m[4]  =  s.y; r.m[8]  =  s.z;
        r.m[1]  =  u.x; r.m[5]  =  u.y; r.m[9]  =  u.z;
        r.m[2]  = -f.x; r.m[6]  = -f.y; r.m[10] = -f.z;
        r.m[12] = -s.dot(eye);
        r.m[13] = -u.dot(eye);
        r.m[14] =  f.dot(eye);
        return r;
    }

    static Mat4 translate(Vec3 t)
    {
        Mat4 r = identity();
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }

    static Mat4 scale(Vec3 s)
    {
        Mat4 r{};
        r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z; r.m[15] = 1.0f;
        return r;
    }

    Mat4 operator*(const Mat4 &b) const
    {
        Mat4 r{};
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 4; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k)
                    sum += m[k * 4 + row] * b.m[c * 4 + k];
                r.m[c * 4 + row] = sum;
            }
        return r;
    }
};

// ─── Shaders ──────────────────────────────────────────────────

static const char *kVertexShader =
    "attribute vec4 aPosition;\n"
    "uniform mat4 uMVP;\n"
    "void main() { gl_Position = uMVP * aPosition; }\n";

static const char *kFragmentShader =
    "precision mediump float;\n"
    "uniform vec4 uColor;\n"
    "void main() { gl_FragColor = uColor; }\n";

// ─── Cube Geometry ────────────────────────────────────────────

static const GLfloat kCubeVerts[] = {
    // Front
    -0.5f,-0.5f, 0.5f,  0.5f,-0.5f, 0.5f,  0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f,
    // Back
    -0.5f,-0.5f,-0.5f, -0.5f, 0.5f,-0.5f,  0.5f, 0.5f,-0.5f,  0.5f,-0.5f,-0.5f,
    // Top
    -0.5f, 0.5f,-0.5f, -0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f,-0.5f,
    // Bottom
    -0.5f,-0.5f,-0.5f,  0.5f,-0.5f,-0.5f,  0.5f,-0.5f, 0.5f, -0.5f,-0.5f, 0.5f,
    // Right
     0.5f,-0.5f,-0.5f,  0.5f, 0.5f,-0.5f,  0.5f, 0.5f, 0.5f,  0.5f,-0.5f, 0.5f,
    // Left
    -0.5f,-0.5f,-0.5f, -0.5f,-0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f,-0.5f,
};

static const GLushort kCubeIdx[] = {
     0, 1, 2,  0, 2, 3,   // Front
     4, 5, 6,  4, 6, 7,   // Back
     8, 9,10,  8,10,11,   // Top
    12,13,14, 12,14,15,   // Bottom
    16,17,18, 16,18,19,   // Right
    20,21,22, 20,22,23,   // Left
};

// ─── Floor geometry ───────────────────────────────────────────

static const GLfloat kFloorVerts[] = {
    -500.f, 0.f, -500.f,
     500.f, 0.f, -500.f,
    -500.f, 0.f,  500.f,
     500.f, 0.f,  500.f,
};

// ─── HSL → RGB helper ─────────────────────────────────────────

static float hue2rgb(float p, float q, float t)
{
    if (t < 0.f) t += 1.f;
    if (t > 1.f) t -= 1.f;
    if (t < 1.f/6.f) return p + (q - p) * 6.f * t;
    if (t < 1.f/2.f) return q;
    if (t < 2.f/3.f) return p + (q - p) * (2.f/3.f - t) * 6.f;
    return p;
}

static void hsl2rgb(float h, float s, float l, float &r, float &g, float &b)
{
    if (s < 0.001f) { r = g = b = l; return; }
    float q = l < 0.5f ? l * (1.f + s) : l + s - l * s;
    float p = 2.f * l - q;
    r = hue2rgb(p, q, h + 1.f/3.f);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1.f/3.f);
}

// ─── App State ────────────────────────────────────────────────

struct AppState {
    struct android_app *app = nullptr;
    Core *core = nullptr;

    // EGL
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLContext context = EGL_NO_CONTEXT;
    int32_t width  = 0;
    int32_t height = 0;

    // GL handles
    GLuint program       = 0;
    GLint  aPosLoc       = -1;
    GLint  uMvpLoc       = -1;
    GLint  uColorLoc     = -1;

    // Connection
    uint32_t myEntityId = 0;
    bool connected      = false;
    bool animating      = false;

    // Camera
    Vec3  camPos   = {0.f, 60.f, -100.f};
    float camYaw   = 0.f;
    float camPitch = -20.f;

    // Touch
    bool  touching   = false;
    float touchX     = 0.f;
    float touchY     = 0.f;

    // Timing
    double lastFrameTime = 0.0;
};

static AppState gState;

// ─── Server IP — change this to your server's WiFi IP ────────

static const char *SERVER_IP   = "172.26.28.251";
static constexpr uint16_t SERVER_PORT = 7777;

// ─── Time helper ──────────────────────────────────────────────

static double nowSeconds()
{
    using Clock = std::chrono::steady_clock;
    static const auto start = Clock::now();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// ─── GL Helpers ───────────────────────────────────────────────

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char buf[512];
        glGetShaderInfoLog(shader, sizeof(buf), nullptr, buf);
        LOGE("Shader compile error: %s", buf);
    }
    return shader;
}

static GLuint createProgram(const char *vs, const char *fs)
{
    GLuint prog = glCreateProgram();
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char buf[512];
        glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        LOGE("Program link error: %s", buf);
    }
    // Shaders can be deleted after linking
    glDeleteShader(v);
    glDeleteShader(f);
    return prog;
}

// ─── EGL Init/Term ────────────────────────────────────────────

static int initDisplay()
{
    const EGLint attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_NONE
    };

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(display, nullptr, nullptr);

    EGLint numConfigs;
    EGLConfig config;
    eglChooseConfig(display, attribs, &config, 1, &numConfigs);

    EGLint format;
    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
    ANativeWindow_setBuffersGeometry(gState.app->window, 0, 0, format);

    EGLSurface surface = eglCreateWindowSurface(display, config, gState.app->window, nullptr);
    EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctxAttribs);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
    {
        LOGE("eglMakeCurrent failed");
        return -1;
    }

    EGLint w, h;
    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    gState.display = display;
    gState.surface = surface;
    gState.context = context;
    gState.width   = w;
    gState.height  = h;

    // Create shader program
    gState.program  = createProgram(kVertexShader, kFragmentShader);
    gState.aPosLoc  = glGetAttribLocation(gState.program, "aPosition");
    gState.uMvpLoc  = glGetUniformLocation(gState.program, "uMVP");
    gState.uColorLoc = glGetUniformLocation(gState.program, "uColor");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.08f, 0.08f, 0.15f, 1.0f);

    LOGI("Display initialized: %dx%d", w, h);
    return 0;
}

static void termDisplay()
{
    if (gState.display != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(gState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (gState.context != EGL_NO_CONTEXT)
            eglDestroyContext(gState.display, gState.context);
        if (gState.surface != EGL_NO_SURFACE)
            eglDestroySurface(gState.display, gState.surface);
        eglTerminate(gState.display);
    }
    gState.animating = false;
    gState.display   = EGL_NO_DISPLAY;
    gState.context   = EGL_NO_CONTEXT;
    gState.surface   = EGL_NO_SURFACE;
}

// ─── Render ───────────────────────────────────────────────────

static void drawFrame()
{
    if (gState.display == EGL_NO_DISPLAY)
        return;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(gState.program);

    // Projection
    float aspect = (gState.height > 0) ? (float)gState.width / (float)gState.height : 1.0f;
    Mat4 proj = Mat4::perspective(45.0f * 3.14159f / 180.0f, aspect, 0.1f, 2000.f);

    // View — follow player entity if connected
    Vec3 eye    = gState.camPos;
    Vec3 center = {eye.x, 0.f, eye.z + 100.f};
    Vec3 up     = {0.f, 1.f, 0.f};

    if (gState.connected && gState.myEntityId != 0 && gState.core)
    {
        int localIdx = -1;
        Partition *chunk = gState.core->world().findEntity(gState.myEntityId, localIdx);
        if (chunk && localIdx >= 0)
        {
            uint32_t readIdx = gState.core->world().getReadIdx();
            auto ent = chunk->getEntity(static_cast<size_t>(localIdx), readIdx);
            Vec3 target = ent.position;
            eye    = {target.x, target.y + 40.f, target.z - 80.f};
            center = target;
        }
    }

    Mat4 view = Mat4::lookAt(eye, center, up);
    Mat4 vp   = proj * view;

    glEnableVertexAttribArray(gState.aPosLoc);

    // ── Draw Floor ───────────────────────────────────────
    glVertexAttribPointer(gState.aPosLoc, 3, GL_FLOAT, GL_FALSE, 0, kFloorVerts);
    glUniform4f(gState.uColorLoc, 0.25f, 0.25f, 0.25f, 1.0f);
    Mat4 floorMvp = vp; // identity model
    glUniformMatrix4fv(gState.uMvpLoc, 1, GL_FALSE, floorMvp.m);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // ── Draw Entities ────────────────────────────────────
    if (!gState.core) goto done;

    glVertexAttribPointer(gState.aPosLoc, 3, GL_FLOAT, GL_FALSE, 0, kCubeVerts);

    {
        uint32_t readIdx = gState.core->world().getReadIdx();
        gState.core->world().forEachChunk([&](Partition &chunk) {
            size_t count = chunk.getEntityCount();
            for (size_t i = 0; i < count; ++i)
            {
                auto ent = chunk.getEntity(i, readIdx);
                uint32_t id = chunk.getEntityId(i);

                // Model = Translate * Scale
                Mat4 model = Mat4::translate(ent.position) * Mat4::scale(ent.size);
                Mat4 mvp   = vp * model;
                glUniformMatrix4fv(gState.uMvpLoc, 1, GL_FALSE, mvp.m);

                // Color: player = green, others = HSL from ID
                float r, g, b;
                if (id == gState.myEntityId)
                {
                    r = 0.1f; g = 1.0f; b = 0.2f;
                }
                else
                {
                    float hue = static_cast<float>(id * 137 % 360) / 360.0f;
                    hsl2rgb(hue, 0.7f, 0.55f, r, g, b);
                }
                glUniform4f(gState.uColorLoc, r, g, b, 1.0f);

                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, kCubeIdx);
            }
        });
    }

done:
    glDisableVertexAttribArray(gState.aPosLoc);
    eglSwapBuffers(gState.display, gState.surface);
}

// ─── Touch → InputManager Mapping ─────────────────────────────

static void updateTouchInput()
{
    if (!gState.core || !gState.connected || gState.myEntityId == 0)
        return;

    auto &im = gState.core->inputManager();
    im.getOrCreate(gState.myEntityId);

    if (!gState.touching)
    {
        // Release all keys when not touching
        im.setKeyState(gState.myEntityId, InputManager::KEY_W, false);
        im.setKeyState(gState.myEntityId, InputManager::KEY_A, false);
        im.setKeyState(gState.myEntityId, InputManager::KEY_S, false);
        im.setKeyState(gState.myEntityId, InputManager::KEY_D, false);
        return;
    }

    // Normalize touch to [-1, 1]
    float nx = (gState.touchX / (float)gState.width)  * 2.0f - 1.0f;
    float ny = (gState.touchY / (float)gState.height) * 2.0f - 1.0f;

    // Deadzone
    constexpr float DEADZONE = 0.15f;

    // Horizontal: left/right → A/D
    im.setKeyState(gState.myEntityId, InputManager::KEY_A, nx < -DEADZONE);
    im.setKeyState(gState.myEntityId, InputManager::KEY_D, nx >  DEADZONE);

    // Vertical: top/bottom → W/S (screen Y is top-down, so ny<0 = top = forward)
    im.setKeyState(gState.myEntityId, InputManager::KEY_W, ny < -DEADZONE);
    im.setKeyState(gState.myEntityId, InputManager::KEY_S, ny >  DEADZONE);
}

// ─── System Registration ──────────────────────────────────────

static void setupSystems(Core &core)
{
    // ── PreSwap ──────────────────────────────────────────

    // 1. Network Receive
    core.registerSystem(Systems::NetworkReceiveSystem(core.network(), core.packetQueue()));

    // 2. Welcome
    core.registerSystem(Systems::WelcomeSystem(core.packetQueue(), gState.myEntityId, gState.connected));

    // 3. State Reconciliation
    core.registerSystem(Systems::StateReconciliationSystem(core.packetQueue()));

    // 4. Touch Input → InputManager
    core.registerSystem({
        "TouchInput", -8,
        [](WorldPartition &/*w*/, float /*dt*/) {
            updateTouchInput();
        },
        {},
        SchedulePhase::PreSwap
    });

    // 5. Spawn local entity if needed
    core.registerSystem({
        "Spawn", -6,
        [](WorldPartition &w, float /*dt*/) {
            if (gState.connected && gState.myEntityId != 0 &&
                !w.isEntityRegistered(gState.myEntityId))
            {
                Partition::EntitySnapshot ent{};
                ent.id = gState.myEntityId;
                ent.position = {0.f, 10.f, 0.f};
                ent.velocity = {0.f, 0.f, 0.f};
                ent.size     = {1.f, 2.f, 1.f};
                ent.health   = 100;
                ent.mass     = 1.f;
                ent.rotation = {0.f, 0.f, 0.f, 1.f};
                w.addEntity(ent);
                LOGI("Spawned local entity %u", gState.myEntityId);
            }
        },
        {{ComponentId::Position, AccessMode::Write}},
        SchedulePhase::PreSwap
    });

    // 6. Movement (client-side prediction)
    core.registerSystem(Systems::MovementSystem(core.inputManager()));

    // 7. Physics
    core.registerSystem(Systems::PhysicsSystem());

    // 8. Input Send (serialize → MSG_INPUTS to server)
    core.registerSystem({
        "InputSend", 15,
        [&core](WorldPartition &/*w*/, float /*dt*/) {
            if (!gState.connected || gState.myEntityId == 0)
                return;

            std::vector<uint8_t> kData, aData, nData;

            // WASD keys
            constexpr uint16_t trackedKeys[] = {
                InputManager::KEY_W, InputManager::KEY_A,
                InputManager::KEY_S, InputManager::KEY_D
            };

            const InputState *st = core.inputManager().getState(gState.myEntityId);
            if (!st) return;

            for (uint16_t key : trackedKeys)
            {
                uint16_t packed = (key & 0x7FFF);
                if (st->getKey(key))
                    packed |= 0x8000;
                kData.push_back(static_cast<uint8_t>(packed & 0xFF));
                kData.push_back(static_cast<uint8_t>((packed >> 8) & 0xFF));
            }

            // No neural data on Android (yet)
            if (!kData.empty())
                core.network().send_inputs(gState.myEntityId, kData, aData, nData);
        },
        {},
        SchedulePhase::PreSwap
    });

    // ── PostSwap ─────────────────────────────────────────

    // 9. Render (reads from read buffer after swap)
    core.registerSystem({
        "Render", 10,
        [](WorldPartition &/*w*/, float /*dt*/) {
            drawFrame();
        },
        {
            {ComponentId::Position, AccessMode::Read},
            {ComponentId::Size,     AccessMode::Read},
            {ComponentId::Health,   AccessMode::Read},
        },
        SchedulePhase::PostSwap
    });

    core.buildSchedule();
    LOGI("ECS schedule built (9 systems)");
}

// ─── Input Handler ────────────────────────────────────────────

static int32_t handleInput(struct android_app * /*app*/, AInputEvent *event)
{
    if (AInputEvent_getType(event) != AINPUT_EVENT_TYPE_MOTION)
        return 0;

    int32_t action = AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;
    switch (action)
    {
        case AMOTION_EVENT_ACTION_DOWN:
        case AMOTION_EVENT_ACTION_MOVE:
            gState.touching = true;
            gState.touchX = AMotionEvent_getX(event, 0);
            gState.touchY = AMotionEvent_getY(event, 0);
            break;
        case AMOTION_EVENT_ACTION_UP:
        case AMOTION_EVENT_ACTION_CANCEL:
            gState.touching = false;
            break;
    }
    return 1;
}

// ─── Lifecycle Handler ────────────────────────────────────────

static void handleCmd(struct android_app *app, int32_t cmd)
{
    switch (cmd)
    {
        case APP_CMD_INIT_WINDOW:
            if (app->window != nullptr)
            {
                initDisplay();
                gState.animating = true;
            }
            break;

        case APP_CMD_TERM_WINDOW:
            termDisplay();
            break;

        case APP_CMD_GAINED_FOCUS:
            gState.animating = true;
            break;

        case APP_CMD_LOST_FOCUS:
            gState.animating = false;
            // Draw one frame to show paused state
            drawFrame();
            break;
    }
}

// ─── android_main ─────────────────────────────────────────────

void android_main(struct android_app *app)
{
    LOGI("=== LplPlugin Android Client ===");

    // Init state
    gState = AppState{};
    gState.app = app;
    app->userData      = &gState;
    app->onAppCmd      = handleCmd;
    app->onInputEvent  = handleInput;

    // Wait for window before creating Core
    // (Core::network_init needs to happen after Android is ready)
    LOGI("Waiting for window...");
    while (!gState.animating)
    {
        int events;
        struct android_poll_source *source;
        while (ALooper_pollAll(100, nullptr, &events, (void**)&source) >= 0)
        {
            if (source) source->process(app, source);
            if (app->destroyRequested) return;
        }
    }

    // Create Core (initializes network in socket mode via -DLPL_USE_SOCKET)
    LOGI("Creating Core...");
    Core core;
    gState.core = &core;

    // Connect to server
    LOGI("Connecting to %s:%d...", SERVER_IP, SERVER_PORT);
    core.initClientNetwork(SERVER_IP, SERVER_PORT);

    // Setup ECS systems
    setupSystems(core);

    // Main loop using Core::runVariableDt
    LOGI("Entering main loop...");
    gState.lastFrameTime = nowSeconds();

    core.runVariableDt(
        // computeDt
        []() -> float {
            double now = nowSeconds();
            float dt = static_cast<float>(now - gState.lastFrameTime);
            gState.lastFrameTime = now;
            if (dt > 0.1f) dt = 0.1f; // Clamp
            return dt;
        },
        // postLoop — process Android events + check destroy
        [&app, &core](float /*dt*/) {
            int events;
            struct android_poll_source *source;
            while (ALooper_pollAll(0, nullptr, &events, (void**)&source) >= 0)
            {
                if (source) source->process(app, source);
                if (app->destroyRequested)
                {
                    core.stop();
                    return;
                }
            }

            // Pause when not animating (window lost)
            if (!gState.animating)
            {
                // Enter a blocking poll until we regain focus or get destroyed
                while (!gState.animating && !app->destroyRequested)
                {
                    ALooper_pollAll(-1, nullptr, &events, (void**)&source);
                    if (source) source->process(app, source);
                }
                if (app->destroyRequested)
                    core.stop();
            }
        }
    );

    // Cleanup
    LOGI("Shutting down...");
    termDisplay();
    gState.core = nullptr;
    LOGI("=== LplPlugin Android Client Terminated ===");
}
