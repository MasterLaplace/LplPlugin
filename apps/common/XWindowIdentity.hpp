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

#    include <cstring>
#    include <unistd.h>

namespace lpl::apps {

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
