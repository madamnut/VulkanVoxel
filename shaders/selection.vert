#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 viewProj;
} ubo;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = ubo.viewProj * vec4(inPosition, 1.0);
}
