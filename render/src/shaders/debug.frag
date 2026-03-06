#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main()
{
    // Simple normal-based diffuse: use world-space normal mapped to [0,1] for
    // easy visual debugging even without a proper light source.
    vec3  N        = normalize(fragNormal);
    float diffuse  = max(dot(N, normalize(vec3(1.0, 2.0, 3.0))), 0.0);
    vec3  baseColor = texture(texSampler, fragUV).rgb;
    outColor = vec4(baseColor * (0.3 + 0.7 * diffuse), 1.0);
}
