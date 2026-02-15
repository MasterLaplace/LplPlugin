// --- LAPLACE TEST CLIENT (ANDROID PORT) --- //
// File: visual_android.cpp
// Description: Client OpenGL ES 2.0 pour Android.
//              Se connecte au serveur LplPlugin (UDP 7777).
//              Utilise android_native_app_glue.

#include <jni.h>
#include <errno.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <android/sensor.h>
#include <android/log.h>
#include <android_native_app_glue.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#include "lpl_protocol.h"
#include "Math.hpp"
#include "WorldPartition.hpp"
#include "Network.hpp"

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "LplVisual", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "LplVisual", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "LplVisual", __VA_ARGS__))

// ─── Utility: vector math substitute (no GLM dependency to keep it simple) ──
struct Mat4 {
    float m[16];
    static Mat4 identity() {
        Mat4 res;
        memset(res.m, 0, sizeof(res.m));
        res.m[0] = res.m[5] = res.m[10] = res.m[15] = 1.0f;
        return res;
    }
    static Mat4 perspective(float fov, float aspect, float n, float f) {
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
    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
        Vec3 f = (center - eye).normalize();
        Vec3 s = f.cross(up).normalize(); // Assuming Vec3 has cross/normalize
        Vec3 u = s.cross(f);
        Mat4 res = identity();
        res.m[0] = s.x; res.m[4] = s.y; res.m[8] = s.z;
        res.m[1] = u.x; res.m[5] = u.y; res.m[9] = u.z;
        res.m[2] = -f.x; res.m[6] = -f.y; res.m[10] = -f.z;
        res.m[12] = -s.dot(eye); res.m[13] = -u.dot(eye); res.m[14] = f.dot(eye);
        return res;
    }
     // Helper for multiplication if needed, but for now we construct V/P directly
};

// ─── Shaders ───
const char* valVertexShader =
    "attribute vec4 vPosition;\n"
    "uniform mat4 uMVP;\n"
    "void main() {\n"
    "  gl_Position = uMVP * vPosition;\n"
    "}\n";

const char* valFragmentShader =
    "precision mediump float;\n"
    "uniform vec4 uColor;\n"
    "void main() {\n"
    "  gl_FragColor = uColor;\n"
    "}\n";

// ─── Game State ───
struct Engine {
    struct android_app* app;

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
};

// ─── Network ───
void init_network(Engine* engine) {
    if (!engine->network.network_init()) {
         LOGE("Network init failed (driver not found)");
         // Assuming Android might fail anyway, we log but continue to not crash immediately?
         // But client is useless without network.
         return;
    }

    // Connect to host (10.0.2.2 emulator alias)
    engine->network.set_server_info("10.0.2.2", 7777);
    engine->network.send_connect("10.0.2.2", 7777);
    LOGI("Sent MSG_CONNECT via Network driver");
}

void process_network(Engine* engine) {
    if (!engine->network.is_connected()) {
        // Poll status or wait for welcome?
        // Actually network_consume_packets handles welcome internally and sets connected/myEntityId in Network class.
        // We just need to sync it.
    }

    engine->network.network_consume_packets(engine->world);

    if (!engine->connected && engine->network.is_connected()) {
        engine->connected = true;
        engine->myEntityId = engine->network.get_local_entity_id();
        LOGI("Connected! EntityID: %d", engine->myEntityId);
    }
}

// ─── Rendering ───
GLuint loadShader(GLenum type, const char* shaderSrc) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);
    return shader;
}

int init_display(Engine* engine) {
    const EGLint attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_NONE
    };
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
    EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    context = eglCreateContext(display, config, NULL, contextAttribs);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
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

void draw_frame(Engine* engine) {
    if (engine->display == NULL) return;

    process_network(engine);

    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(engine->program);

    // Setup Matrices
    Mat4 proj = Mat4::perspective(45.0f * 3.14159f / 180.0f, (float)engine->width/engine->height, 0.1f, 1000.f);
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
    GLfloat vertices[] = {
        -500.f, 0.f, -500.f,
         500.f, 0.f, -500.f,
        -500.f, 0.f,  500.f,
         500.f, 0.f,  500.f
    };

    glVertexAttribPointer(engine->vPositionHandle, 3, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(engine->vPositionHandle);

    // Pass Color
    glUniform4f(engine->uColorHandle, 0.3f, 0.3f, 0.3f, 1.0f);

    // Pass Matrix (Identity for now to verify shader works)
    GLfloat identity[16];
    memset(identity, 0, sizeof(identity));
    identity[0] = identity[5] = identity[10] = identity[15] = 1.0f;
    // Scale down to see valid range
    identity[0] = 0.001f; identity[5] = 0.001f;

    glUniformMatrix4fv(engine->uMVPHandle, 1, GL_FALSE, identity);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    eglSwapBuffers(engine->display, engine->surface);
}

void term_display(Engine* engine) {
    if (engine->display != EGL_NO_DISPLAY) {
        eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (engine->context != EGL_NO_CONTEXT) {
            eglDestroyContext(engine->display, engine->context);
        }
        if (engine->surface != EGL_NO_SURFACE) {
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
int32_t handle_input(struct android_app* app, AInputEvent* event) {
    Engine* engine = (Engine*)app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
        // Touch input: simple move forward command
        if (engine->connected) {
             engine->network.send_input(engine->myEntityId, {0.f, 0.f, 1.f});
        }
        return 1;
    }
    return 0;
}

// ─── Lifecycle ───
void handle_cmd(struct android_app* app, int32_t cmd) {
    Engine* engine = (Engine*)app->userData;
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (engine->app->window != NULL) {
                init_display(engine);
                draw_frame(engine);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            term_display(engine);
            break;
        case APP_CMD_GAINED_FOCUS:
            engine->animating = true;
            break;
        case APP_CMD_LOST_FOCUS:
            engine->animating = false;
            draw_frame(engine);
            break;
    }
}

void android_main(struct android_app* state) {
    Engine engine;
    memset(&engine, 0, sizeof(engine));
    state->userData = &engine;
    state->onAppCmd = handle_cmd;
    state->onInputEvent = handle_input;
    engine.app = state;

    init_network(&engine);

    while (1) {
        int ident;
        int events;
        struct android_poll_source* source;

        while ((ident = ALooper_pollAll(engine.animating ? 0 : -1, NULL, &events, (void**)&source)) >= 0) {
            if (source != NULL) {
                source->process(state, source);
            }
            if (state->destroyRequested != 0) {
                term_display(&engine);
                return;
            }
        }

        if (engine.animating) {
            draw_frame(&engine);
        }
    }
}
