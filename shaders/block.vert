#version 450

layout(set = 0, binding = 0) uniform BlockUniform
{
    mat4 viewProjection;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUv;
layout(location = 2) in float inAo;
layout(location = 3) in float inTextureLayer;
layout(location = 0) out vec2 outUv;
layout(location = 1) out float outAo;
layout(location = 2) out float outTextureLayer;

void main()
{
    outUv = inUv;
    outAo = inAo;
    outTextureLayer = inTextureLayer;
    gl_Position = ubo.viewProjection * vec4(inPosition, 1.0);
}
