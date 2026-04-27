#version 450

layout(set = 0, binding = 1) uniform sampler2D sunTexture;
layout(set = 0, binding = 2) uniform sampler2D moonTexture;

layout(location = 0) in vec2 inUv;
layout(location = 1) flat in int inSpriteIndex;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 color = inSpriteIndex == 0 ? texture(sunTexture, inUv) : texture(moonTexture, inUv);
    if (color.a < 0.05)
    {
        discard;
    }

    outColor = color;
}

