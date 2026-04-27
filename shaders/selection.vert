#version 450

layout(set = 0, binding = 0) uniform BlockUniform
{
    mat4 viewProjection;
} ubo;

layout(location = 0) in vec3 inPosition;

void main()
{
    gl_Position = ubo.viewProjection * vec4(inPosition, 1.0);
}
