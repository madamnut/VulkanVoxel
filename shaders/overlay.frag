#version 450

layout(binding = 1) uniform sampler2D fontAtlas;

layout(location = 0) in vec2 fragUv;
layout(location = 1) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 sampleColor = texture(fontAtlas, fragUv);
    if (sampleColor.a <= 0.001) {
        discard;
    }

    outColor = vec4(sampleColor.rgb * fragColor, sampleColor.a);
}
