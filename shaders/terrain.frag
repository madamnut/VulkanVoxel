#version 450

layout(binding = 1) uniform sampler2DArray blockTexture;

layout(location = 0) in vec2 fragUv;
layout(location = 1) in float fragAo;
layout(location = 2) flat in uint fragTextureLayer;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 color = texture(blockTexture, vec3(fragUv, float(fragTextureLayer)));
    outColor = vec4(color.rgb * fragAo, color.a);
}
