#version 450

layout(set = 0, binding = 1) uniform sampler2D playerTexture;

layout(location = 0) in vec2 inUv;
layout(location = 1) in float inAo;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 color = texture(playerTexture, inUv);
    if (color.a < 0.01)
    {
        discard;
    }
    outColor = vec4(color.rgb * inAo, color.a);
}
