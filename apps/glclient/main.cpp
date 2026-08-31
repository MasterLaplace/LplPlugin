/**
 * @file main.cpp
 * @brief The ring-0 client, on a desktop, through a window.
 *
 * @warning **This app contains no rendering.** It opens a window, hands the engine a framebuffer,
 * and uploads whatever the engine put there. Every pixel is drawn by the same
 * `engine::TerrainRenderer` running the same `procgen` passes over the same authoritative Fixed32
 * state that the kernel runs -- so what appears here is what QEMU shows, at desktop speed and with
 * a mouse. Drawing the world a second time in OpenGL would be faster and would be a second
 * renderer: two answers to what the world looks like, free to disagree, and the one that is
 * checked by five booted artifacts would not be this one.
 *
 * OpenGL is therefore a BLITTER and nothing else -- one textured quad. That is also why this costs
 * so little: `samples::TerrainWorld` rasterises at 480x300 and scales up, so the host pays for an
 * upload, not for a scene.
 *
 * @warning **It also required no engine code.** `platform::IDisplayBackend` and
 * `platform::IInputBackend` were declared for exactly this and had one implementation each -- a
 * kernel scanout and a host stub that drew into RAM nobody looked at. `engine::bootGame` already
 * took a platform and a World factory. The only thing missing was somebody to plug a window into
 * the seam, which is what this file is.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

// @warning **The engine headers come FIRST, and the order is not cosmetic.** Xlib defines `None`,
// `Bool`, `Status` and `Success` as bare preprocessor macros, and `history::Predicate::None` is an
// enumerator by that name -- so including X11 first turns an enum member into `0L` and the error
// points at Predicate.hpp, a file this app never edits. Every project that touches both hits this
// once; hitting it in the right order costs nothing.
#include <lpl/core/Log.hpp>
#include <lpl/engine/Boot.hpp>
#include <lpl/pack/ViewerPackBlob.hpp>
#include <lpl/platform/linux/LinuxPlatform.hpp>
#include <lpl/samples/ChronicleWorld.hpp>
#include <lpl/samples/TerrainWorld.hpp>
#include <lpl/std/memory.hpp>

#include "../common/XWindowIdentity.hpp"

#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/XKBlib.h>
#include <X11/Xlib.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace lpl;

namespace {

using lpl::apps::centreOn;
using lpl::apps::MonitorRect;
using lpl::apps::primaryMonitor;

/**
 * @class DesktopHost
 * @brief The X11 window, its GL context, the framebuffer the engine draws into, and the events.
 *
 * @warning One object rather than two, because the display and the input are the SAME connection:
 * a window that pumped its own events and an input backend that pumped its own would each drain
 * the other's, and the symptom is a keyboard that works only on the frames the mouse is still.
 * The two seams below are adapters over this; nothing owns an X connection twice.
 */
class DesktopHost {
public:
    /**
     * @brief Opens the window and the GL context.
     *
     * @param width  Framebuffer width in pixels.
     * @param height Framebuffer height in pixels.
     * @return false when there is no display, or no usable visual.
     */
    /**
     * @brief Puts the window on a specific monitor instead of the primary one.
     *
     * @warning Indexed the way the server enumerates them, and reported by --diagnose, so the
     * number a user passes is a number they were shown rather than one they had to guess.
     *
     * @param index Monitor ordinal, or a negative value to keep the primary.
     */
    /**
     * @brief Puts the window's top-left corner at an exact screen position.
     *
     * @warning Screen coordinates, which on a multi-monitor X screen span every display: the
     * layout --diagnose prints is the map to read them against. Overrides the monitor choice,
     * because a caller who names a corner has already decided.
     *
     * @param x Left edge.
     * @param y Top edge.
     */
    void placeAt(int x, int y) noexcept
    {
        _explicitX = x;
        _explicitY = y;
        _hasExplicitCorner = true;
    }

    /** @brief Asks for fullscreen on whichever monitor the window lands on. */
    void requestFullscreen() noexcept { _wantFullscreen = true; }

    void placeOnMonitor(int index)
    {
        if (index < 0)
            return;
        Display *probe = XOpenDisplay(nullptr);
        if (probe == nullptr)
            return;
        int count = 0;
        XRRMonitorInfo *monitors = XRRGetMonitors(probe, DefaultRootWindow(probe), True, &count);
        if (monitors != nullptr && index < count)
        {
            _placement.x = monitors[index].x;
            _placement.y = monitors[index].y;
            _placement.width = monitors[index].width;
            _placement.height = monitors[index].height;
            _placement.fromRandr = true;
        }
        if (monitors != nullptr)
            XRRFreeMonitors(monitors);
        XCloseDisplay(probe);
    }

    [[nodiscard]] bool open(core::u32 width, core::u32 height)
    {
        _width = width;
        _height = height;
        // @warning **Seeded from the requested size, not left at one.** These used to be set only by
        // ConfigureNotify, so until that event arrived the viewport was one pixel by one -- and a
        // window manager is under no obligation to send a configure for a window that opened at
        // the size it asked for. The result is a window that maps, swaps buffers, and shows the
        // grey it was born with, which looks exactly like a GL context that failed to bind.
        _windowWidth = static_cast<int>(width);
        _windowHeight = static_cast<int>(height);
        _pixels.assign(static_cast<std::size_t>(width) * height, 0u);

        _display = XOpenDisplay(nullptr);
        if (_display == nullptr)
        {
            std::fprintf(stderr, "lpl-glclient: cannot open a display (is DISPLAY set?)\n");
            return false;
        }

        // No depth buffer asked for: the engine owns the depth test and does it in software, in
        // the same buffer the kernel uses. A GL depth buffer here would be an unused attachment.
        int attributes[] = {GLX_RGBA, GLX_DOUBLEBUFFER, None};
        _visual = glXChooseVisual(_display, DefaultScreen(_display), attributes);
        if (_visual == nullptr)
        {
            std::fprintf(stderr, "lpl-glclient: no double-buffered RGBA visual\n");
            return false;
        }

        Window root = DefaultRootWindow(_display);
        XSetWindowAttributes windowAttributes{};
        windowAttributes.colormap = XCreateColormap(_display, root, _visual->visual, AllocNone);
        windowAttributes.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask |
                                      ButtonReleaseMask | PointerMotionMask | StructureNotifyMask;

        // @warning **Centred on the PRIMARY monitor, not at the origin.** An X screen spanning two
        // displays is one coordinate space, so (0,0) is whichever monitor owns that corner -- on a
        // vertically stacked pair that is routinely the second one. The server is asked rather
        // than guessed at.
        const MonitorRect monitor = _placement.width > 0 ? _placement : primaryMonitor(_display);
        int originX = 0;
        int originY = 0;
        centreOn(monitor, static_cast<int>(width), static_cast<int>(height), originX, originY);
        if (_hasExplicitCorner)
        {
            originX = _explicitX;
            originY = _explicitY;
        }

        _window =
            XCreateWindow(_display, root, originX, originY, static_cast<unsigned>(width), static_cast<unsigned>(height),
                          0, _visual->depth, InputOutput, _visual->visual, CWColormap | CWEventMask, &windowAttributes);
        XStoreName(_display, _window, "lpl-glclient - the ring-0 world, on a desktop");
        // Before the map, and it is what decides whether anything is ever shown under a RAIL
        // compositor. See XWindowIdentity.hpp for the log that proves it.
        apps::declareWindowIdentity(_display, _window, "lpl-glclient", "LplGlclient", static_cast<int>(width),
                                    static_cast<int>(height), originX, originY);
        if (_wantFullscreen)
            apps::requestFullscreen(_display, _window);
        XMapWindow(_display, _window);

        // Drawing into a window the server has not mapped yet is drawing into nothing, and the
        // frames are simply lost. Waiting for the map costs one blocking read at startup and
        // removes a race whose symptom is a window that stays blank for as long as the machine
        // happens to be slow.
        XEvent mapped;
        XIfEvent(
            _display, &mapped,
            [](Display *, XEvent *event, XPointer argument) {
                return static_cast<Bool>(event->type == MapNotify && event->xmap.window == *(Window *) argument);
            },
            (XPointer) &_window);

        // The window manager's close button arrives as a client message, and a window that
        // ignores it is a window that only Ctrl-C closes.
        _closeAtom = XInternAtom(_display, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(_display, _window, &_closeAtom, 1);

        // @warning Without this, held keys do not exist: X synthesises a KeyRelease immediately
        // before every auto-repeat KeyPress, so a key held down reads as released on most frames
        // and the walk stutters instead of moving. Detectable auto-repeat suppresses the
        // synthetic release, which is the whole reason `isKeyHeld` can be answered honestly.
        Bool supported = False;
        XkbSetDetectableAutoRepeat(_display, True, &supported);
        _detectableRepeat = supported == True;

        _context = glXCreateContext(_display, _visual, nullptr, GL_TRUE);
        glXMakeCurrent(_display, _window, _context);

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_BLEND);
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &_texture);
        glBindTexture(GL_TEXTURE_2D, _texture);
        // NEAREST, deliberately. The engine rasterises at 480x300 and scales up itself; a second
        // bilinear pass here would blur the upscale the world already chose, and the point of this
        // app is to show the frame the kernel produces rather than a smoothed version of it.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_BGRA,
                     GL_UNSIGNED_BYTE, _pixels.data());

        // @warning **Painted before anything else happens.** The window is created at the top of
        // main and the first frame cannot arrive until a world has been generated -- which on a
        // debug build is seconds of procgen passes and four least-cost searches. Until then the
        // window shows whatever the compositor left in it, which is a grey rectangle, and a grey
        // rectangle is indistinguishable from a GL context that failed. One clear and one swap
        // costs nothing and makes the difference visible.
        glClearColor(0.02f, 0.03f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glXSwapBuffers(_display, _window);
        glClear(GL_COLOR_BUFFER_BIT);
        glXSwapBuffers(_display, _window); // both buffers, or the first real swap flashes the grey back
        return true;
    }

    /** @brief Tears down the context and the window. */
    void close()
    {
        if (_display == nullptr)
            return;
        if (_texture != 0u)
            glDeleteTextures(1, &_texture);
        glXMakeCurrent(_display, None, nullptr);
        if (_context != nullptr)
            glXDestroyContext(_display, _context);
        if (_window != 0)
            XDestroyWindow(_display, _window);
        XCloseDisplay(_display);
        _display = nullptr;
    }

    /**
     * @brief Drains the X queue into the input rings.
     *
     * @warning Called from present(), which is the only point in the frame the host is
     * guaranteed to reach. Pumping from the input backend instead would tie event delivery to
     * whether the World happened to ask for a key this frame.
     */
    void pump()
    {
        while (XPending(_display) > 0)
        {
            XEvent event;
            XNextEvent(_display, &event);
            switch (event.type)
            {
            case ClientMessage:
                if (static_cast<Atom>(event.xclient.data.l[0]) == _closeAtom)
                    _shouldClose = true;
                break;
            case ConfigureNotify:
                _windowWidth = event.xconfigure.width;
                _windowHeight = event.xconfigure.height;
                break;
            case KeyPress:
            case KeyRelease: {
                char text[8]{};
                KeySym symbol = 0;
                const int written = XLookupString(&event.xkey, text, sizeof(text) - 1u, &symbol, nullptr);
                if (symbol == XK_Escape)
                {
                    _shouldClose = true;
                    break;
                }
                if (written <= 0)
                    break;
                const auto code = static_cast<unsigned char>(text[0]);
                if (code >= sizeof(_held))
                    break;
                if (event.type == KeyPress)
                {
                    // Only the edge is queued. A held key already reports through isKeyHeld, and
                    // queueing every repeat as well would stack a typed nudge on top of a steady
                    // walk -- which is the lurch TerrainWorld's own comment describes.
                    if (_held[code] == 0u)
                        pushCharacter(text[0]);
                    _held[code] = 1u;
                }
                else
                {
                    _held[code] = 0u;
                }
                break;
            }
            case ButtonPress:
            case ButtonRelease: {
                const unsigned button = event.xbutton.button;
                const core::u32 mask = button == Button1 ? 1u : button == Button3 ? 2u : button == Button2 ? 4u : 0u;
                if (event.type == ButtonPress)
                    _buttons |= mask;
                else
                    _buttons &= ~mask;
                break;
            }
            case MotionNotify:
                // Deltas against the previous position, which is what the seam asks for: an
                // absolute position would make the world's look speed depend on the window size.
                if (_haveMotion)
                    pushMotion(event.xmotion.x - _lastX, event.xmotion.y - _lastY);
                _lastX = event.xmotion.x;
                _lastY = event.xmotion.y;
                _haveMotion = true;
                break;
            default: break;
            }
        }
    }

    /** @brief Uploads the framebuffer and swaps. */
    void present()
    {
        pump();
        if (_display == nullptr)
            return;

        glViewport(0, 0, _windowWidth, _windowHeight);
        glBindTexture(GL_TEXTURE_2D, _texture);
        // BGRA over a 0x00RRGGBB word: on a little-endian host those bytes are B, G, R, 0, which
        // is exactly what GL_BGRA reads. Alpha comes out zero and nothing blends, by design.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(_width), static_cast<GLsizei>(_height), GL_BGRA,
                        GL_UNSIGNED_BYTE, _pixels.data());

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0); // y down: row 0 of the framebuffer is the top row
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(0.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(1.0f, 0.0f);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(0.0f, 1.0f);
        glEnd();

        // @warning **Read back BEFORE the swap, from the buffer that is about to become the window.**
        // The self-check used to count colours in `_pixels`, which is what the ENGINE wrote -- so a
        // window showing nothing at all passed it, and did: the viewport was one pixel by one and
        // the check reported twenty-eight colours. An instrument on the wrong side of the seam it
        // is meant to test is worse than none, because it reads as evidence.
        const bool lastFrame = _frameBudget != 0u && _framesPresented + 1u >= _frameBudget;
        const bool heartbeat = _diagnoseEvery != 0u && (_framesPresented % _diagnoseEvery) == 0u;
        if (lastFrame || heartbeat)
        {
            _presented.assign(static_cast<std::size_t>(_windowWidth) * static_cast<std::size_t>(_windowHeight), 0u);
            glReadBuffer(GL_BACK);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, _windowWidth, _windowHeight, GL_BGRA, GL_UNSIGNED_BYTE, _presented.data());
        }

        if (heartbeat)
        {
            // @warning Reported from INSIDE the presentation path, because there is no other way to
            // see it: an external screenshot tool is not always there, and the framebuffer the
            // engine wrote says nothing about what reached the window.
            unsigned realWidth = 0u;
            unsigned realHeight = 0u;
            Window root = 0;
            int originX = 0;
            int originY = 0;
            unsigned border = 0u;
            unsigned depth = 0u;
            XGetGeometry(_display, _window, &root, &originX, &originY, &realWidth, &realHeight, &border, &depth);
            std::printf("glclient: frame %u  window %ux%u  viewport %dx%d  colours on screen %u\n", _framesPresented,
                        realWidth, realHeight, _windowWidth, _windowHeight, distinctIn(_presented));
            std::fflush(stdout);
        }

        glXSwapBuffers(_display, _window);
        if (_framesPresented == 0u)
        {
            // The line that separates "still generating" from "running but blank". Without it the
            // log's last word is about the real-time guard, and a reader cannot tell whether the
            // window ever received a frame.
            std::printf("glclient: first frame on screen\n");
            std::fflush(stdout);
        }
        ++_framesPresented;
        if (_frameBudget != 0u && _framesPresented >= _frameBudget)
            _shouldClose = true;
    }

    [[nodiscard]] core::u32 *pixels() noexcept { return _pixels.data(); }
    [[nodiscard]] core::u32 width() const noexcept { return _width; }
    [[nodiscard]] core::u32 height() const noexcept { return _height; }
    [[nodiscard]] bool shouldClose() const noexcept { return _shouldClose; }
    [[nodiscard]] bool detectableRepeat() const noexcept { return _detectableRepeat; }

    [[nodiscard]] bool isHeld(char character) const noexcept
    {
        const auto code = static_cast<unsigned char>(character);
        return code < sizeof(_held) && _held[code] != 0u;
    }

    /**
     * @brief Takes one queued character.
     *
     * @param out Receives it.
     * @return false when the ring is empty.
     */
    [[nodiscard]] bool popCharacter(char &out)
    {
        if (_charHead == _charTail)
            return false;
        out = _chars[_charTail];
        _charTail = (_charTail + 1u) & (kRing - 1u);
        return true;
    }

    [[nodiscard]] core::u32 pendingCharacters() const noexcept { return (_charHead - _charTail) & (kRing - 1u); }

    /**
     * @brief Takes one accumulated pointer motion.
     *
     * @param outDeltaX Receives the horizontal delta.
     * @param outDeltaY Receives the vertical delta.
     * @param outButtons Receives the button mask.
     * @return false when nothing moved.
     */
    [[nodiscard]] bool popMotion(core::i32 &outDeltaX, core::i32 &outDeltaY, core::u32 &outButtons)
    {
        if (_motionHead == _motionTail)
            return false;
        outDeltaX = _motionX[_motionTail];
        outDeltaY = _motionY[_motionTail];
        outButtons = _buttons;
        _motionTail = (_motionTail + 1u) & (kRing - 1u);
        ++_motionsTaken;
        return true;
    }

    [[nodiscard]] core::u32 motionsTaken() const noexcept { return _motionsTaken; }

    /**
     * @brief Stops after @p frames presents, or never when zero.
     *
     * @warning Exists so this app can be checked without somebody looking at it. A window nobody
     * asserts on is how the viewers in this tree drifted: the failure of a render path is a black
     * rectangle, which is indistinguishable from a window that has not drawn yet.
     *
     * @param frames How many frames to present before asking to close.
     */
    void limitFrames(core::u32 frames) noexcept { _frameBudget = frames; }

    /**
     * @brief Reports the geometry and the presented colour count every @p frames frames.
     *
     * @param frames How often, or zero to stay quiet.
     */
    void diagnoseEvery(core::u32 frames) noexcept { _diagnoseEvery = frames; }

    /** @brief Prints what the GL implementation says it is. */
    void reportContext() const
    {
        lpl::apps::reportMonitors(_display);
        std::printf("glclient: GL %s | %s | %s | %s rendering\n",
                    reinterpret_cast<const char *>(glGetString(GL_VERSION)),
                    reinterpret_cast<const char *>(glGetString(GL_VENDOR)),
                    reinterpret_cast<const char *>(glGetString(GL_RENDERER)),
                    glXIsDirect(_display, _context) == True ? "direct" : "INDIRECT");
        std::fflush(stdout);
    }

    [[nodiscard]] core::u32 framesPresented() const noexcept { return _framesPresented; }

    /**
     * @brief Writes the framebuffer as a binary PPM.
     *
     * @param path Destination.
     * @return false when the file could not be written.
     */
    [[nodiscard]] bool writePortablePixmap(const char *path) const
    {
        std::FILE *file = std::fopen(path, "wb");
        if (file == nullptr)
            return false;
        std::fprintf(file, "P6\n%u %u\n255\n", _width, _height);
        for (std::size_t i = 0u; i < _pixels.size(); ++i)
        {
            const core::u32 pixel = _pixels[i];
            const unsigned char rgb[3] = {static_cast<unsigned char>((pixel >> 16) & 0xFFu),
                                          static_cast<unsigned char>((pixel >> 8) & 0xFFu),
                                          static_cast<unsigned char>(pixel & 0xFFu)};
            std::fwrite(rgb, 1u, 3u, file);
        }
        std::fclose(file);
        return true;
    }

    /**
     * @brief How many distinct colours the last frame holds.
     *
     * @warning The one number that separates "the world drew" from "the window is a colour". A
     * black frame and a cleared frame both have one; a landscape has thousands.
     *
     * @return The count, capped at the sample size.
     */
    [[nodiscard]] core::u32 distinctColours() const { return distinctIn(_pixels); }

    /**
     * @brief Colours in the frame the WINDOW received, rather than the one the engine drew.
     *
     * @warning The two are different questions and only this one is about the presentation path.
     * @return The count, or zero when no frame was read back.
     */
    [[nodiscard]] core::u32 distinctPresentedColours() const { return distinctIn(_presented); }

    /** @brief Whether a frame was read back off the GL buffer at all. */
    [[nodiscard]] bool capturedPresented() const noexcept { return !_presented.empty(); }

    [[nodiscard]] static core::u32 distinctIn(const std::vector<core::u32> &pixels)
    {
        std::vector<core::u32> seen;
        seen.reserve(4096u);
        for (std::size_t i = 0u; i < pixels.size(); i += 7u) // a prime stride: no row alignment
        {
            const core::u32 pixel = pixels[i] & 0x00FFFFFFu;
            bool known = false;
            for (const core::u32 other : seen)
                if (other == pixel)
                {
                    known = true;
                    break;
                }
            if (!known)
            {
                seen.push_back(pixel);
                if (seen.size() >= 4096u)
                    break;
            }
        }
        return static_cast<core::u32>(seen.size());
    }

private:
    static constexpr core::u32 kRing = 256u; ///< Power of two: the wrap is a mask.

    void pushCharacter(char character)
    {
        const core::u32 next = (_charHead + 1u) & (kRing - 1u);
        if (next == _charTail)
            return; // full: drop the newest rather than corrupt the oldest, as the kernel ring does
        _chars[_charHead] = character;
        _charHead = next;
    }

    void pushMotion(int deltaX, int deltaY)
    {
        const core::u32 next = (_motionHead + 1u) & (kRing - 1u);
        if (next == _motionTail)
            return;
        _motionX[_motionHead] = deltaX;
        _motionY[_motionHead] = deltaY;
        _motionHead = next;
    }

    Display *_display{nullptr};
    XVisualInfo *_visual{nullptr};
    Window _window{0};
    GLXContext _context{nullptr};
    Atom _closeAtom{0};
    GLuint _texture{0u};

    core::u32 _width{0u};
    core::u32 _height{0u};
    int _windowWidth{1};
    int _windowHeight{1};
    std::vector<core::u32> _pixels;
    std::vector<core::u32> _presented;

    bool _shouldClose{false};
    bool _detectableRepeat{false};

    core::u8 _held[128]{};
    char _chars[kRing]{};
    core::u32 _charHead{0u};
    core::u32 _charTail{0u};

    core::i32 _motionX[kRing]{};
    core::i32 _motionY[kRing]{};
    core::u32 _motionHead{0u};
    core::u32 _motionTail{0u};
    core::u32 _motionsTaken{0u};
    core::u32 _buttons{0u};
    core::u32 _frameBudget{0u};
    core::u32 _diagnoseEvery{0u};
    MonitorRect _placement{};
    int _explicitX{0};
    int _explicitY{0};
    bool _hasExplicitCorner{false};
    bool _wantFullscreen{false};
    core::u32 _framesPresented{0u};
    int _lastX{0};
    int _lastY{0};
    bool _haveMotion{false};
};

/** @brief The display seam over the window. */
class DesktopDisplay final : public platform::IDisplayBackend {
public:
    explicit DesktopDisplay(DesktopHost &host) noexcept : _host(host) {}

    [[nodiscard]] bool querySurface(platform::SurfaceDescriptor &outDescriptor) const noexcept override
    {
        outDescriptor.buffer = const_cast<DesktopHost &>(_host).pixels();
        outDescriptor.physicalAddress = 0u; // no GPU uploader attaches to this one
        outDescriptor.width = _host.width();
        outDescriptor.height = _host.height();
        outDescriptor.pitch = _host.width() * 4u;
        outDescriptor.bitsPerPixel = 32u;
        return outDescriptor.buffer != nullptr;
    }

    void clear(core::u32 colorRgb) override
    {
        core::u32 *pixels = _host.pixels();
        const core::u32 count = _host.width() * _host.height();
        for (core::u32 i = 0u; i < count; ++i)
            pixels[i] = colorRgb;
    }

    [[nodiscard]] core::u32 readPixel(core::u32 x, core::u32 y) const noexcept override
    {
        if (x >= _host.width() || y >= _host.height())
            return 0u;
        return const_cast<DesktopHost &>(_host).pixels()[y * _host.width() + x];
    }

    void present() override { _host.present(); }
    [[nodiscard]] bool shouldClose() const noexcept override { return _host.shouldClose(); }
    [[nodiscard]] const char *name() const noexcept override { return "DesktopDisplay(GLX blit)"; }

private:
    DesktopHost &_host;
};

/** @brief The input seam over the same window. */
class DesktopInput final : public platform::IInputBackend {
public:
    explicit DesktopInput(DesktopHost &host) noexcept : _host(host) {}

    [[nodiscard]] bool tryPopCharacter(char &outCharacter) override { return _host.popCharacter(outCharacter); }
    [[nodiscard]] core::u32 pendingCount() const noexcept override { return _host.pendingCharacters(); }

    [[nodiscard]] bool tryPopPointerMotion(core::i32 &outDeltaX, core::i32 &outDeltaY, core::u32 &outButtons) override
    {
        return _host.popMotion(outDeltaX, outDeltaY, outButtons);
    }

    [[nodiscard]] bool isKeyHeld(char character) const noexcept override { return _host.isHeld(character); }

    // @warning Answered honestly rather than unconditionally: without detectable auto-repeat X
    // reports a release before every repeat, so held keys would be wrong most frames. Saying so
    // lets the World fall back to its typed path instead of walking in stutters.
    [[nodiscard]] bool hasKeyStates() const noexcept override { return _host.detectableRepeat(); }
    [[nodiscard]] bool hasPointer() const noexcept override { return true; }
    [[nodiscard]] core::u32 pointerInterruptCount() const noexcept override { return _host.motionsTaken(); }
    [[nodiscard]] const char *name() const noexcept override { return "DesktopInput(X11)"; }

private:
    DesktopHost &_host;
};

/**
 * @class DesktopPlatform
 * @brief LinuxPlatform's clock and memory, with a real window in front.
 *
 * @warning It reuses `LinuxClockBackend`, `LinuxMemoryBackend` and `LinuxGpuMemoryBackend` rather
 * than restating them: what a desktop adds is a display and a keyboard, and a second host clock
 * would be a second answer to what time it is.
 */
class DesktopPlatform final : public platform::IPlatform {
public:
    explicit DesktopPlatform(DesktopHost &host) noexcept : _display(host), _input(host) {}

    [[nodiscard]] platform::IClockBackend &clock() noexcept override { return _clock; }
    [[nodiscard]] platform::IDisplayBackend &display() noexcept override { return _display; }
    [[nodiscard]] platform::IInputBackend &input() noexcept override { return _input; }
    [[nodiscard]] platform::IMemoryBackend &memory() noexcept override { return _memory; }
    [[nodiscard]] platform::IGpuMemoryBackend &gpuMemory() noexcept override { return _gpuMemory; }
    [[nodiscard]] const char *name() const noexcept override { return "DesktopPlatform(X11/GLX)"; }

private:
    platform::linux_host::LinuxClockBackend _clock;
    DesktopDisplay _display;
    DesktopInput _input;
    platform::linux_host::LinuxMemoryBackend _memory;
    platform::linux_host::LinuxGpuMemoryBackend _gpuMemory;
};

} // namespace

int main(int argc, char **argv)
{
    core::u32 width = 1280u;
    core::u32 height = 800u;
    core::u32 frames = 0u;
    bool chronicle = false;
    core::u32 diagnose = 0u;
    int monitor = -1;
    int atX = 0;
    int atY = 0;
    bool hasCorner = false;
    bool fullscreen = false;
    const char *shot = nullptr;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
        {
            frames = static_cast<core::u32>(std::atoi(argv[i + 1]));
            ++i;
            continue;
        }
        if (std::strcmp(argv[i], "--at") == 0 && i + 2 < argc)
        {
            atX = std::atoi(argv[i + 1]);
            atY = std::atoi(argv[i + 2]);
            hasCorner = true;
            i += 2;
            continue;
        }
        if (std::strcmp(argv[i], "--fullscreen") == 0)
        {
            fullscreen = true;
            continue;
        }
        if (std::strcmp(argv[i], "--monitor") == 0 && i + 1 < argc)
        {
            monitor = std::atoi(argv[i + 1]);
            ++i;
            continue;
        }
        if (std::strcmp(argv[i], "--diagnose") == 0)
        {
            diagnose = 60u;
            continue;
        }
        if (std::strcmp(argv[i], "--chronicle") == 0)
        {
            chronicle = true;
            continue;
        }
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc)
        {
            shot = argv[i + 1];
            ++i;
            continue;
        }
        if (std::strcmp(argv[i], "--size") == 0 && i + 2 < argc)
        {
            width = static_cast<core::u32>(std::atoi(argv[i + 1]));
            height = static_cast<core::u32>(std::atoi(argv[i + 2]));
            i += 2;
        }
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("usage: lpl-glclient [--size W H] [--frames N] [--shot out.ppm]\n"
                        "  Runs the same World the kernel client boots, in a window.\n"
                        "  WASD walk, mouse looks, space jumps, V toggles the body, Escape quits.\n"
                        "  --frames N quits after N frames and --shot writes the last one, so the\n"
                        "  app can be checked without anybody looking at it.\n"
                        "  --chronicle runs ChronicleWorld instead: the Mani over a century, with\n"
                        "  the roads a corpus attests and the people who walked them.\n"
                        "  --diagnose reports the monitors, the GL context, the real window size\n"
                        "  and how many colours actually reached the screen, every sixty frames.\n"
                        "  --monitor N opens on that monitor instead of the primary one; the\n"
                        "  numbering is the one --diagnose prints.\n"
                        "  --at X Y puts the top-left corner at an exact screen coordinate, which\n"
                        "  on a multi-monitor X screen spans every display.\n"
                        "  --fullscreen asks the window manager for fullscreen on that monitor.\n");
            return 0;
        }
    }

    DesktopHost host;
    host.placeOnMonitor(monitor);
    if (hasCorner)
        host.placeAt(atX, atY);
    if (fullscreen)
        host.requestFullscreen();
    if (!host.open(width, height))
        return 1;
    host.limitFrames(frames);
    host.diagnoseEvery(diagnose);
    if (diagnose != 0u)
        host.reportContext();

    engine::BootRequest request;
    request.host = engine::HostProfile::DesktopClient;
    request.tickRate = 60u;
    request.banner = "=== LplPlugin GL Client ===";
    // The SAME built-in cartridge the kernel client falls back to, so the desktop and ring 0 show
    // the same world rather than two worlds that happen to share a renderer.
    request.fallbackPackBytes = pack::kViewerPackBytes;
    request.fallbackPackSize = pack::kViewerPackSize;

    const engine::BootResult result = engine::bootGame(
        request, pmr::unique_ptr<platform::IPlatform>{new DesktopPlatform{host}},
        [chronicle](const procgen::WorldRecipe &recipe, const ecology::LivingRecipe &living,
                    const engine::ViewProfile &view) {
            if (chronicle)
                return pmr::unique_ptr<engine::World>{pmr::make_unique<samples::ChronicleWorld>(recipe)};
            return pmr::unique_ptr<engine::World>{pmr::make_unique<samples::TerrainWorld>(recipe, living, view)};
        });

    const core::u32 colours = host.distinctColours();
    const core::u32 presented = host.distinctPresentedColours();
    // @warning Zero has to be distinguished from "never looked". The readback fires on the last
    // budgeted frame, so a run cut short never captures anything -- and printing 0 for that reads
    // as a black window, which is the one thing this line exists to detect. Two meanings on one
    // number is the defect this session kept finding; it does not get to live here.
    if (host.capturedPresented())
        std::printf("glclient: %u frames presented, %u colours drawn, %u colours actually on screen\n",
                    host.framesPresented(), colours, presented);
    else
        std::printf("glclient: %u frames presented, %u colours drawn, screen not sampled "
                    "(the run ended before the budgeted frame)\n",
                    host.framesPresented(), colours);
    if (shot != nullptr && !host.writePortablePixmap(shot))
        core::Log::error("glclient: could not write the screenshot");

    host.close();
    // A frame that is one colour is a render path that did not run, and it looks exactly like a
    // window that has not drawn yet. Reported as a failure rather than a clean exit.
    if (frames != 0u && colours < 2u)
    {
        std::fprintf(stderr, "lpl-glclient: the last frame is a single colour\n");
        return 2;
    }
    // @warning The one that matters, and the one that was missing: the engine can draw a perfect
    // frame into a buffer the window never receives. That is not a hypothetical -- it is what a
    // one-pixel viewport did, while the check above reported twenty-eight colours.
    if (frames != 0u && host.capturedPresented() && presented < 2u)
    {
        std::fprintf(stderr, "lpl-glclient: the window received a single colour - the frame was drawn "
                             "and not presented\n");
        return 3;
    }
    return result.initialised ? 0 : 1;
}
