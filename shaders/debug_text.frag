#version 450

layout(set = 0, binding = 0) uniform sampler2D debugTextTexture;

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main()
{
    float alpha = texture(debugTextTexture, inUv).a * inColor.a;
    if (alpha < 0.01)
    {
        discard;
    }

    outColor = vec4(inColor.rgb, alpha);
}
