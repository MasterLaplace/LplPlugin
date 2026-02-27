#include "gui/GUI.hpp"

namespace lpl::render::vk {

void GUI::Render(const glm::vec4 &clear_color, const VkCommandBuffer &commandBuffer)
{
    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (_show_demo_window)
        ImGui::ShowDemoWindow(&_show_demo_window);

    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");           // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &_show_demo_window); // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &_show_another_window);

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);              // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float *) &clear_color); // Edit 3 floats representing a color

        if (ImGui::Button(
                "Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / _io->Framerate, _io->Framerate);
        ImGui::End();
    }

    if (_show_another_window)
    {
        ImGui::Begin("Another Window",
                     &_show_another_window); // Pass a pointer to our bool variable (the window will have a closing
                                             // button that will clear the bool when clicked)
        ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
            _show_another_window = false;
        ImGui::End();
    }

    ImGui::Render();
    ImDrawData *draw_data = ImGui::GetDrawData();
    const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
    if (!is_minimized)
    {
        ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
    }
}

void GUI::check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}

GUI::GUI(const CreateInfo &info)
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    _io = &ImGui::GetIO();
    _io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    _io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(info.window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = info.instance;
    init_info.PhysicalDevice = info.physicalDevice;
    init_info.Device = info.device;
    init_info.QueueFamily = info.queueFamily;
    init_info.Queue = info.queue;
    init_info.PipelineCache = _pipelineCache;
    init_info.DescriptorPool = info.descriptorPool;
    init_info.PipelineInfoMain.RenderPass = info.renderPass;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.MinImageCount = 2; // >= 2
    init_info.ImageCount = 2; // >= MinImageCount
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = info.allocator;
    init_info.CheckVkResultFn = GUI::check_vk_result;
    ImGui_ImplVulkan_Init(&init_info);
}

GUI::~GUI()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

} // namespace lpl::render::vk
