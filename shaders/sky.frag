#version 450

layout(binding = 0) uniform SkyUniformBufferObject {
    mat4 viewProj;
    vec4 cameraRight;
    vec4 cameraUp;
    vec4 cameraForward;
    vec4 projectionParams;
} ubo;

layout(location = 0) in vec2 fragUv;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 ndc = fragUv * 2.0 - 1.0;
    vec3 viewRay = normalize(
        ubo.cameraForward.xyz +
        ubo.cameraRight.xyz * ndc.x * ubo.projectionParams.x +
        ubo.cameraUp.xyz * ndc.y * ubo.projectionParams.y
    );

    float t = clamp(viewRay.y * 0.5 + 0.5, 0.0, 1.0);
    t = smoothstep(0.0, 1.0, t);

    vec3 horizonColor = vec3(0.78, 0.89, 1.00);
    vec3 zenithColor = vec3(0.45, 0.68, 0.98);
    vec3 skyColor = mix(horizonColor, zenithColor, t);

    outColor = vec4(skyColor, 1.0);
}
