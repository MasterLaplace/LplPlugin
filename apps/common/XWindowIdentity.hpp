/**
 * @file XWindowIdentity.hpp
 * @brief What a raw-Xlib window has to say about itself before anybody will show it.
 *
 * @warning **The bug this exists to prevent, found under WSLg and confirmed from the compositor's
 * own log.** Weston in RAIL mode -- which is how WSLg presents Linux windows as Windows windows --
 * derives an application identity from `WM_CLASS`. Xlib does not set it for you. Every toolkit
 * does (GTK, Qt, SDL, GLFW), so nobody writing against one ever meets this; two apps in this tree
 * open their windows by hand and neither set it.
 *
 * The symptom is not a crash and not an error. The window is created, mapped, and reported
 * `IsViewable` by the server; it gets a generic taskbar entry whose tooltip shows the right title;
 * and it never appears on any screen, and clicking the taskbar entry does nothing. The only place
 * the cause is stated is the compositor:
 *
 *     ClientGetAppidReq: WindowId:0x9 does not have appId, or not top level window
 *
 * and with the hints below, on the same machine, the same run:
 *
 *     associateWindowId: 1   appId: Lplwin   appDesc: Lplwin
 *     ClientGetAppidReq: pid:0 appId:Lplwin WindowId:0xe
 *
 * @warning Shared rather than copied into each app, because the two that need it are exactly the
 * two that have already drifted apart once: `mapview`'s creature loop diverged from the engine's
 * in both directions while nobody was compiling it. A fix applied to one window bootstrap and not
 * the other is that story again.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_APPS_XWINDOWIDENTITY_HPP
#    define LPL_APPS_XWINDOWIDENTITY_HPP

#    include <X11/Xatom.h>
#    include <X11/Xlib.h>
#    include <X11/Xutil.h>
#    include <X11/extensions/Xrandr.h>

#    include <cstdio>
#    include <cstring>
#    include <unistd.h>

namespace lpl::apps {

/**
 * @struct MonitorRect
 * @brief One monitor's place in the X screen, in screen coordinates.
 */
struct MonitorRect {
    int x{0};             ///< Left edge within the X screen.
    int y{0};             ///< Top edge within the X screen.
    int width{0};         ///< Width in pixels.
    int height{0};        ///< Height in pixels.
    bool primary{false};  ///< Whether the server calls this one primary.
    bool fromRandr{false};///< False when this is the whole-screen fallback.
};

/**
 * @brief The monitor the server calls primary, or the whole screen if it will not say.
 *
 * @warning **RandR and not Xinerama**, for one reason: Xinerama enumerates rectangles and has no
 * concept of a primary one, so a caller using it has to guess -- and "the first one" is not the
 * same monitor on every machine. An X screen spanning two displays is one coordinate space, so
 * a window created at (0,0) lands on whichever monitor happens to own that corner, which on a
 * vertically stacked pair is routinely the wrong one.
 *
 * @param display The connection.
 * @return The primary monitor, or the full screen when RandR reports nothing.
 */
[[nodiscard]] inline MonitorRect primaryMonitor(Display *display)
{
    MonitorRect chosen;
    const int screen = DefaultScreen(display);
    chosen.width = DisplayWidth(display, screen);
    chosen.height = DisplayHeight(display, screen);

    int count = 0;
    // get_active true: a monitor that is configured and switched off is not somewhere to put a
    // window, and RandR will happily list it.
    XRRMonitorInfo *monitors = XRRGetMonitors(display, RootWindow(display, screen), True, &count);
    if (monitors == nullptr || count <= 0)
    {
        if (monitors != nullptr)
            XRRFreeMonitors(monitors);
        return chosen;
    }

    // @warning **A server may mark NO monitor primary, and then index zero is a coin toss.**
    // Measured under WSLg: two monitors, `rdp-0 1920x1200 at (0,1440)` and `rdp-2 2560x1440 at
    // (0,0)`, and neither carries the primary flag -- so taking the first put the window on the
    // lower screen, which is exactly the one the user was not looking at.
    //
    // The fallback is the monitor that owns the ORIGIN, because that is where X conventionally
    // puts the main display, with area as the tie-break for a layout that does not cover it. Both
    // rules pick rdp-2 here, and they agree for the same reason: the screen everything else is
    // measured from is the screen somebody is sitting in front of.
    int pick = -1;
    for (int i = 0; i < count; ++i)
        if (monitors[i].primary != 0)
        {
            pick = i;
            break;
        }
    for (int i = 0; pick < 0 && i < count; ++i)
        if (monitors[i].x == 0 && monitors[i].y == 0)
            pick = i;
    if (pick < 0)
    {
        pick = 0;
        for (int i = 1; i < count; ++i)
            if (static_cast<long>(monitors[i].width) * monitors[i].height >
                static_cast<long>(monitors[pick].width) * monitors[pick].height)
                pick = i;
    }

    chosen.x = monitors[pick].x;
    chosen.y = monitors[pick].y;
    chosen.width = monitors[pick].width;
    chosen.height = monitors[pick].height;
    chosen.primary = monitors[pick].primary != 0;
    chosen.fromRandr = true;
    XRRFreeMonitors(monitors);
    return chosen;
}

/**
 * @brief Prints every monitor the server reports.
 *
 * @warning A diagnostic, and it earns its place: "the window opened on the wrong screen" cannot be
 * acted on without knowing what the server thinks the screens ARE, and an X screen spanning two
 * displays looks like one big desktop from inside the client.
 *
 * @param display The connection.
 */
inline void reportMonitors(Display *display)
{
    const int screen = DefaultScreen(display);
    int count = 0;
    XRRMonitorInfo *monitors = XRRGetMonitors(display, RootWindow(display, screen), True, &count);
    std::printf("monitors: X screen %dx%d, RandR reports %d\n", DisplayWidth(display, screen),
                DisplayHeight(display, screen), count);
    for (int i = 0; i < count; ++i)
    {
        char *name = XGetAtomName(display, monitors[i].name);
        std::printf("  [%d] %s %dx%d at (%d,%d)%s\n", i, name != nullptr ? name : "?", monitors[i].width,
                    monitors[i].height, monitors[i].x, monitors[i].y,
                    monitors[i].primary != 0 ? "  PRIMARY" : "");
        if (name != nullptr)
            XFree(name);
    }
    std::fflush(stdout);
    if (monitors != nullptr)
        XRRFreeMonitors(monitors);
}

/**
 * @brief Top-left corner that centres a window of @p width by @p height on @p monitor.
 *
 * @param monitor Where to put it.
 * @param width   Window width.
 * @param height  Window height.
 * @param outX    Receives the X position.
 * @param outY    Receives the Y position.
 */
inline void centreOn(const MonitorRect &monitor, int width, int height, int &outX, int &outY)
{
    outX = monitor.x + (monitor.width - width) / 2;
    outY = monitor.y + (monitor.height - height) / 2;
    // A window taller or wider than its monitor would centre to a negative corner, which puts its
    // title bar off the top where nothing can grab it.
    if (outX < monitor.x)
        outX = monitor.x;
    if (outY < monitor.y)
        outY = monitor.y;
}

/**
 * @brief Declares the window's identity, size and input hints.
 *
 * @warning **Call this BEFORE XMapWindow.** A window manager reads these when the window is
 * mapped; setting them afterwards leaves the association already made without them, and under
 * RAIL that association is the one that decides whether anything is ever drawn.
 *
 * @param display      The connection.
 * @param window       The window, created and not yet mapped.
 * @param instanceName WM_CLASS instance -- conventionally the executable's name.
 * @param className    WM_CLASS class -- conventionally the same, capitalised. This is the one
 *                     RAIL turns into an appId.
 * @param width        Width the window was created with.
 * @param height       Height the window was created with.
 * @param x            X position it was created at.
 * @param y            Y position it was created at.
 */
inline void declareWindowIdentity(Display *display, Window window, const char *instanceName, const char *className,
                                  int width, int height, int x = 0, int y = 0)
{
    XClassHint classHint;
    // Xlib's fields are char* rather than const char*: the cast is Xlib's age, not a lie about
    // ownership -- XSetClassHint copies both strings into the server's property.
    classHint.res_name = const_cast<char *>(instanceName);
    classHint.res_class = const_cast<char *>(className);
    XSetClassHint(display, window, &classHint);

    XSizeHints sizeHints;
    std::memset(&sizeHints, 0, sizeof(sizeHints));
    sizeHints.flags = PPosition | PSize;
    sizeHints.x = x;
    sizeHints.y = y;
    sizeHints.width = width;
    sizeHints.height = height;
    XSetWMNormalHints(display, window, &sizeHints);

    XWMHints wmHints;
    std::memset(&wmHints, 0, sizeof(wmHints));
    wmHints.flags = InputHint | StateHint;
    wmHints.input = True;
    wmHints.initial_state = NormalState;
    XSetWMHints(display, window, &wmHints);

    // _NET_WM_PID lets the compositor tie the window to a process rather than to a name alone.
    // Not what RAIL keys on, but it is what makes the association survive two windows sharing a
    // class -- and the log prints `pid:0` without it.
    const Atom pidAtom = XInternAtom(display, "_NET_WM_PID", False);
    const long pid = static_cast<long>(getpid());
    XChangeProperty(display, window, pidAtom, XA_CARDINAL, 32, PropModeReplace,
                    reinterpret_cast<const unsigned char *>(&pid), 1);

    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0)
    {
        hostname[sizeof(hostname) - 1] = '\0';
        XTextProperty machine;
        char *machineName = hostname;
        if (XStringListToTextProperty(&machineName, 1, &machine) != 0)
        {
            XSetWMClientMachine(display, window, &machine);
            XFree(machine.value);
        }
    }
}

} // namespace lpl::apps

#endif // LPL_APPS_XWINDOWIDENTITY_HPP
