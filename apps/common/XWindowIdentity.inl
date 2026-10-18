#ifndef LPL_APPS_XWINDOWIDENTITY_INL
#define LPL_APPS_XWINDOWIDENTITY_INL

namespace lpl::apps {

inline MonitorRect primaryMonitor(Display *display)
{
    MonitorRect chosen;
    const int screen = DefaultScreen(display);
    chosen.width = DisplayWidth(display, screen);
    chosen.height = DisplayHeight(display, screen);

    int count = 0;
    XRRMonitorInfo *monitors = XRRGetMonitors(display, RootWindow(display, screen), True, &count);
    if (monitors == nullptr || count <= 0)
    {
        if (monitors != nullptr)
            XRRFreeMonitors(monitors);
        return chosen;
    }

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
                    monitors[i].height, monitors[i].x, monitors[i].y, monitors[i].primary != 0 ? "  PRIMARY" : "");
        if (name != nullptr)
            XFree(name);
    }
    std::fflush(stdout);
    if (monitors != nullptr)
        XRRFreeMonitors(monitors);
}

inline void centreOn(const MonitorRect &monitor, int width, int height, int &outX, int &outY)
{
    outX = monitor.x + (monitor.width - width) / 2;
    outY = monitor.y + (monitor.height - height) / 2;
    if (outX < monitor.x)
        outX = monitor.x;
    if (outY < monitor.y)
        outY = monitor.y;
}

inline void declareWindowIdentity(Display *display, Window window, const char *instanceName, const char *className,
                                  int width, int height, int x, int y)
{
    XClassHint classHint;
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

inline void requestFullscreen(Display *display, Window window)
{
    const Atom state = XInternAtom(display, "_NET_WM_STATE", False);
    const Atom fullscreen = XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False);
    XChangeProperty(display, window, state, XA_ATOM, 32, PropModeReplace,
                    reinterpret_cast<const unsigned char *>(&fullscreen), 1);
}

} // namespace lpl::apps

#endif // LPL_APPS_XWINDOWIDENTITY_INL
