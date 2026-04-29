#version 450

layout(set = 0, binding = 1) uniform sampler2DArray blockTexture;

layout(location = 0) in vec2 inUv;
layout(location = 1) in float inAo;
layout(location = 2) in float inTextureLayer;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 color = texture(blockTexture, vec3(fract(inUv), inTextureLayer));
    if (color.a < 0.5)
    {
        discard;
    }
    outColor = vec4(color.rgb * inAo, color.a);
}
