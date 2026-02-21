// --- LAPLACE TEST CLIENT (ANDROID PORT) --- //
// File: visual_android.cpp
// Description: Client OpenGL ES 2.0 pour Android.
//              Se connecte au serveur LplPlugin (UDP 7777).
//              Utilise android_native_app_glue.

#include <android/log.h>
#include <android/sensor.h>
#include <android_native_app_glue.h>
#include <cmath>
#include <cstring>
#include <errno.h>
#include <jni.h>
#include <vector>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "Math.hpp"
#include "Network.hpp"
#include "WorldPartition.hpp"
#include "lpl_protocol.h"

#define LOGI(...) ((void) __android_log_print(ANDROID_LOG_INFO, "LplVisual", __VA_ARGS__))
#define LOGW(...) ((void) __android_log_print(ANDROID_LOG_WARN, "LplVisual", __VA_ARGS__))
#define LOGE(...) ((void) __android_log_print(ANDROID_LOG_ERROR, "LplVisual", __VA_ARGS__))

// ─── Utility: vector math substitute (no GLM dependency to keep it simple) ──
struct Mat4 {
    float m[16];
    static Mat4 identity()
    {
        Mat4 res;
        memset(res.m, 0, sizeof(res.m));
        res.m[0] = res.m[5] = res.m[10] = res.m[15] = 1.0f;
        return res;
    }
    static Mat4 perspective(float fov, float aspect, float n, float f)
    {
        Mat4 res;
        memset(res.m, 0, sizeof(res.m));
        float tanHalfFov = tanf(fov / 2.0f);
        res.m[0] = 1.0f / (aspect * tanHalfFov);
        res.m[5] = 1.0f / (tanHalfFov);
        res.m[10] = -(f + n) / (f - n);
        res.m[11] = -1.0f;
        res.m[14] = -(2.0f * f * n) / (f - n);
        return res;
    }
    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up)
    {
        Vec3 f = (center - eye).normalize();
        Vec3 s = f.cross(up).normalize(); // Assuming Vec3 has cross/normalize
        Vec3 u = s.cross(f);
        Mat4 res = identity();
        res.m[0] = s.x;
        res.m[4] = s.y;
        res.m[8] = s.z;
        res.m[1] = u.x;
        res.m[5] = u.y;
        res.m[9] = u.z;
        res.m[2] = -f.x;
        res.m[6] = -f.y;
        res.m[10] = -f.z;
        res.m[12] = -s.dot(eye);
        res.m[13] = -u.dot(eye);
        res.m[14] = f.dot(eye);
        return res;
    }
    // Helper for multiplication if needed, but for now we construct V/P directly
};

// ─── Shaders ───
const char *valVertexShader = "attribute vec4 vPosition;\n"
                              "uniform mat4 uMVP;\n"
                              "void main() {\n"
                              "  gl_Position = uMVP * vPosition;\n"
                              "}\n";

const char *valFragmentShader = "precision mediump float;\n"
                                "uniform vec4 uColor;\n"
                                "void main() {\n"
                                "  gl_FragColor = uColor;\n"
                                "}\n";

// ─── Game State ───
struct Engine {
    struct android_app *app;

    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
    int32_t width;
    int32_t height;

    bool animating;
    bool connected;
    int sock;
    struct sockaddr_in serverAddr;
    uint32_t myEntityId;

    // OpenGL handles
    GLuint program;
    GLuint vPositionHandle;
    GLuint uMVPHandle;
    GLuint uColorHandle;

    // Camera
    Vec3 camPos;
    float camYaw;

    // World
    WorldPartition world;
    Network network;
    bool localTest;
};

// ─── Network ───
void init_network(Engine *engine)
{
    if (!engine->network.network_init())
    {
        LOGE("Network init failed (driver not found)");
        // Assuming Android might fail anyway, we log but continue to not crash immediately?
        // But client is useless without network.
        return;
    }

    // Connect to host (LAN IP)
    engine->network.set_server_info("10.35.10.1", 7777);
    engine->network.send_connect("10.35.10.1", 7777);
    LOGI("Sent MSG_CONNECT via Network driver");
}

void process_network(Engine *engine)
{
    if (!engine->network.is_connected())
    {
        // Poll status or wait for welcome?
        // Actually network_consume_packets handles welcome internally and sets connected/myEntityId in Network class.
        // We just need to sync it.
    }

    engine->network.network_consume_packets(engine->world);

    if (!engine->connected && engine->network.is_connected())
    {
        engine->connected = true;
        engine->myEntityId = engine->network.get_local_entity_id();
        LOGI("Connected! EntityID: %d", engine->myEntityId);
    }
}

// ─── Rendering ───
GLuint loadShader(GLenum type, const char *shaderSrc)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);
    return shader;
}

int init_display(Engine *engine)
{
    const EGLint attribs[] = {EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8,
                              EGL_NONE};
    EGLint w, h, dummy, format;
    EGLint numConfigs;
    EGLConfig config;
    EGLSurface surface;
    EGLContext context;

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(display, 0, 0);
    eglChooseConfig(display, attribs, &config, 1, &numConfigs);
    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);

    ANativeWindow_setBuffersGeometry(engine->app->window, 0, 0, format);

    surface = eglCreateWindowSurface(display, config, engine->app->window, NULL);
    EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    context = eglCreateContext(display, config, NULL, contextAttribs);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
    {
        LOGW("Unable to eglMakeCurrent");
        return -1;
    }

    engine->display = display;
    engine->context = context;
    engine->surface = surface;
    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);
    engine->width = w;
    engine->height = h;

    // Load Shaders
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, valVertexShader);
    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, valFragmentShader);
    engine->program = glCreateProgram();
    glAttachShader(engine->program, vertexShader);
    glAttachShader(engine->program, fragmentShader);
    glLinkProgram(engine->program);

    engine->vPositionHandle = glGetAttribLocation(engine->program, "vPosition");
    engine->uColorHandle = glGetUniformLocation(engine->program, "uColor");
    engine->uMVPHandle = glGetUniformLocation(engine->program, "uMVP");

    glEnable(GL_DEPTH_TEST);
    return 0;
}

void draw_frame(Engine *engine)
{
    if (engine->display == NULL)
        return;

    if (!engine->localTest)
        process_network(engine);

    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(engine->program);

    // Setup Matrices
    Mat4 proj = Mat4::perspective(45.0f * 3.14159f / 180.0f, (float) engine->width / engine->height, 0.1f, 1000.f);
    // Simple orbit cam
    Vec3 eye = {0, 50, -100};
    Vec3 center = {0, 0, 0};
    Vec3 up = {0, 1, 0};
    Mat4 view = Mat4::lookAt(eye, center, up);

    // MVP (ignoring model transform for now, just identity)
    // Mult: Proj * View * Model
    // (Simplification: just manually mul P*V for now)
    // Warning: standard matrix mul order...

    // Draw Floor Grid (Simplified as a single quad)
    GLfloat floorVertices[] = {-500.f, 0.f, -500.f, 500.f, 0.f, -500.f, -500.f, 0.f, 500.f, 500.f, 0.f, 500.f};

    glVertexAttribPointer(engine->vPositionHandle, 3, GL_FLOAT, GL_FALSE, 0, floorVertices);
    glEnableVertexAttribArray(engine->vPositionHandle);

    // Pass Color (Grey Floor)
    glUniform4f(engine->uColorHandle, 0.3f, 0.3f, 0.3f, 1.0f);

    // Pass Matrix (Identity model for floor)
    glUniformMatrix4fv(engine->uMVPHandle, 1, GL_FALSE, (proj * view).m);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // --- Draw Entities ---
    static const GLfloat cubeVertices[] = {
        // Front face
        -0.5f,
        -0.5f,
        0.5f,
        0.5f,
        -0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        -0.5f,
        0.5f,
        0.5f,
        // Back face
        -0.5f,
        -0.5f,
        -0.5f,
        -0.5f,
        0.5f,
        -0.5f,
        0.5f,
        0.5f,
        -0.5f,
        0.5f,
        -0.5f,
        -0.5f,
        // Top face
        -0.5f,
        0.5f,
        -0.5f,
        -0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        -0.5f,
        // Bottom face
        -0.5f,
        -0.5f,
        -0.5f,
        0.5f,
        -0.5f,
        -0.5f,
        0.5f,
        -0.5f,
        0.5f,
        -0.5f,
        -0.5f,
        0.5f,
        // Right face
        0.5f,
        -0.5f,
        -0.5f,
        0.5f,
        0.5f,
        -0.5f,
        0.5f,
        0.5f,
        0.5f,
        0.5f,
        -0.5f,
        0.5f,
        // Left face
        -0.5f,
        -0.5f,
        -0.5f,
        -0.5f,
        -0.5f,
        0.5f,
        -0.5f,
        0.5f,
        0.5f,
        -0.5f,
        0.5f,
        -0.5f,
    };

    static const GLushort cubeIndices[] = {
        0,  1,  2,  0,  2,  3,  // Front
        4,  5,  6,  4,  6,  7,  // Back
        8,  9,  10, 8,  10, 11, // Top
        12, 13, 14, 12, 14, 15, // Bottom
        16, 17, 18, 16, 18, 19, // Right
        20, 21, 22, 20, 22, 23  // Left
    };

    glVertexAttribPointer(engine->vPositionHandle, 3, GL_FLOAT, GL_FALSE, 0, cubeVertices);

    uint32_t readIdx = engine->world.getReadIdx();
    Mat4 vp = proj * view; // Precompute ViewProj

    engine->world.forEachChunk([&](Partition &chunk) {
        size_t count = chunk.getEntityCount();
        Vec3 *positions = chunk.getPositionsData(readIdx);
        Vec3 *sizes = chunk.getSizesData(); // Need to expose getSizesData in Partition class or use public accessor
                                            // Accessing cold data directly if friendly, else use accessor
                                            // Partition.hpp: _sizes is private but we have setSize/getSize... wait
                                            // We need direct access for performance or just use getEntity loop
    });

    // Re-check Partition.hpp for accessors.
    // It has `getSizesData`?
    // Checking previous view_file of Partition.hpp...
    // Line 250 in view: `CUDA_CHECK(cudaHostGetDevicePointer(&d_masses, partition.getMassesData(), 0));`
    // It has `getMassesData`, `getHealthData`.
    // Does it have `getSizesData`?
    // Line 650: `std::vector<Vec3, PinnedAllocator<Vec3>> _sizes;`
    // I need to add `getSizesData()` to Partition.hpp if it's missing, OR use `getEntity(i)` which returns a struct
    // with `size`.

    engine->world.forEachChunk([&](Partition &chunk) {
        size_t count = chunk.getEntityCount();
        for (size_t i = 0; i < count; ++i)
        {
            auto entity = chunk.getEntity(i, readIdx);

            // Simple Model Matrix: Scale * Translate (Simpler than full 4x4 mul for just cubes)
            // But we need to construct it.
            // M = T * S

            Vec3 pos = entity.position;
            Vec3 size = entity.size;

            Mat4 model = Mat4::identity();
            model.m[0] = size.x;
            model.m[5] = size.y;
            model.m[10] = size.z;
            model.m[12] = pos.x;
            model.m[13] = pos.y;
            model.m[14] = pos.z;

            Mat4 mvp = vp * model;

            // Color based on entity ID to see individual instances
            float r = (float) ((1000 + i) * 123 % 255) / 255.f;
            float g = (float) ((1000 + i) * 456 % 255) / 255.f;
            float b = (float) ((1000 + i) * 789 % 255) / 255.f;
            glUniform4f(engine->uColorHandle, r, g, b, 1.0f);

            glUniformMatrix4fv(engine->uMVPHandle, 1, GL_FALSE, mvp.m);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, cubeIndices);
        }
    });

    eglSwapBuffers(engine->display, engine->surface);
}

void term_display(Engine *engine)
{
    if (engine->display != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (engine->context != EGL_NO_CONTEXT)
        {
            eglDestroyContext(engine->display, engine->context);
        }
        if (engine->surface != EGL_NO_SURFACE)
        {
            eglDestroySurface(engine->display, engine->surface);
        }
        eglTerminate(engine->display);
    }
    engine->animating = false;
    engine->display = EGL_NO_DISPLAY;
    engine->context = EGL_NO_CONTEXT;
    engine->surface = EGL_NO_SURFACE;
}

// ─── Input ───
int32_t handle_input(struct android_app *app, AInputEvent *event)
{
    Engine *engine = (Engine *) app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
    {
        // Touch input: simple move forward command
        if (engine->connected && !engine->localTest)
        {
            engine->network.send_input(engine->myEntityId, {0.f, 0.f, 1.f});
        }
        return 1;
    }
    return 0;
}

// ─── Lifecycle ───
void handle_cmd(struct android_app *app, int32_t cmd)
{
    Engine *engine = (Engine *) app->userData;
    switch (cmd)
    {
    case APP_CMD_INIT_WINDOW:
        if (engine->app->window != NULL)
        {
            init_display(engine);
            draw_frame(engine);
        }
        break;
    case APP_CMD_TERM_WINDOW: term_display(engine); break;
    case APP_CMD_GAINED_FOCUS: engine->animating = true; break;
    case APP_CMD_LOST_FOCUS:
        engine->animating = false;
        draw_frame(engine);
        break;
    }
}

void android_main(struct android_app *state)
{
    Engine engine;
    memset(&engine, 0, sizeof(engine));
    state->userData = &engine;
    state->onAppCmd = handle_cmd;
    state->onInputEvent = handle_input;
    engine.app = state;

    // --- LOCAL TEST MODE ---
    engine.localTest = true;

    if (!engine.localTest)
    {
        init_network(&engine);
    }
    else
    {
        LOGI("LOCAL TEST MODE: Skipping Network. Spawning entities...");
        // Spawn some cubes
        for (int i = 0; i < 50; ++i)
        {
            Partition::EntitySnapshot ent{};
            ent.id = 1000 + i;
            ent.position = {(float) (i % 5) * 2.0f, 20.0f + (i / 5) * 5.0f, 0.0f};
            ent.size = {1.f, 1.f, 1.f};
            ent.mass = 1.0f;
            ent.health = 100;
            ent.rotation = {0, 0, 0, 1};
            engine.world.addEntity(ent);
        }
        engine.connected = true; // Fake connection to allow input
        engine.myEntityId = 1000;
    }

    while (1)
    {
        int ident;
        int events;
        struct android_poll_source *source;

        while ((ident = ALooper_pollAll(engine.animating ? 0 : -1, NULL, &events, (void **) &source)) >= 0)
        {
            if (source != NULL)
            {
                source->process(state, source);
            }
            if (state->destroyRequested != 0)
            {
                term_display(&engine);
                return;
            }
        }

        if (engine.animating)
        {
            if (engine.localTest)
            {
                // Run local physics
                engine.world.step(0.016f);
                engine.world.swapBuffers();
            }
            draw_frame(&engine);
        }
    }
}
