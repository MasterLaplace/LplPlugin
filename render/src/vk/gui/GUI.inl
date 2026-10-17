#ifndef LPL_RENDER_VK_GUI_INL_
#define LPL_RENDER_VK_GUI_INL_

namespace lpl::render::vk {

inline void GUI::CreateInstance(const CreateInfo &info)
{
    if (_instance == nullptr)
        _instance = std::make_unique<GUI>(info);
}

inline void GUI::DestroyInstance()
{
    if (_instance != nullptr)
        _instance.reset();
}

inline GUI &GUI::GetInstance()
{
    if (!_instance)
    {
        ::lpl::core::Log::fatal("GUI instance not created!");
        std::abort();
    }
    return *_instance;
}

} // namespace lpl::render::vk

#endif // LPL_RENDER_VK_GUI_INL_
