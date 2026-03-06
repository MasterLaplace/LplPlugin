#version 450

// Vertex attributes (must match GraphicsPipeline.cpp attribute descriptions)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

// UBO (binding 0) matches UniformBufferObject in UniformObject.hpp
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;

void main()
{
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    // Transform normal to view space for simple ambient+diffuse shading
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragUV     = inUV;
}
