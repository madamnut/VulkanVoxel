#version 450

layout(binding = 1) uniform sampler2D blockTexture;

layout(location = 0) in vec2 fragUv;
layout(location = 1) in float fragAo;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 color = texture(blockTexture, fragUv);
    outColor = vec4(color.rgb * fragAo, color.a);
}
