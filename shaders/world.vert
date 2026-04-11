#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 viewProj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUv;
layout(location = 2) in float inAo;

layout(location = 0) out vec2 fragUv;
layout(location = 1) out float fragAo;

void main() {
    gl_Position = ubo.viewProj * vec4(inPosition, 1.0);
    fragUv = inUv;
    fragAo = inAo;
}
