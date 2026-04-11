#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 viewProj;
} ubo;

layout(push_constant) uniform TerrainPushConstants {
    int chunkMinX;
    int chunkMinZ;
} pc;

layout(location = 0) in uint inPacked0;
layout(location = 1) in uint inPacked1;

layout(location = 0) out vec2 fragUv;
layout(location = 1) out float fragAo;

float DecodeAo(uint value) {
    const float brightnessTable[4] = float[4](0.55, 0.70, 0.85, 1.0);
    return brightnessTable[int(value & 0x3u)];
}

int GetCornerIndex(int localVertexIndex, bool flipDiagonal, bool reverseWinding) {
    if (flipDiagonal) {
        if (reverseWinding) {
            const int indices[6] = int[6](0, 3, 1, 1, 3, 2);
            return indices[localVertexIndex];
        }

        const int indices[6] = int[6](0, 1, 3, 1, 2, 3);
        return indices[localVertexIndex];
    }

    if (reverseWinding) {
        const int indices[6] = int[6](0, 2, 1, 0, 3, 2);
        return indices[localVertexIndex];
    }

    const int indices[6] = int[6](0, 1, 2, 0, 2, 3);
    return indices[localVertexIndex];
}

void main() {
    int localVertexIndex = gl_VertexIndex % 6;

    uint localX = inPacked0 & 0xFu;
    uint localY = (inPacked0 >> 4u) & 0xFu;
    uint localZ = (inPacked0 >> 8u) & 0xFu;
    uint subChunkIndex = (inPacked0 >> 12u) & 0x1Fu;
    uint axis = (inPacked0 >> 17u) & 0x3u;
    bool positiveNormal = ((inPacked0 >> 19u) & 0x1u) != 0u;
    uint width = ((inPacked0 >> 20u) & 0xFu) + 1u;
    uint height = ((inPacked0 >> 24u) & 0xFu) + 1u;
    bool flipDiagonal = ((inPacked0 >> 28u) & 0x1u) != 0u;
    bool reverseWinding = ((inPacked0 >> 29u) & 0x1u) != 0u;

    int uAxis = int((axis + 1u) % 3u);
    int vAxis = int((axis + 2u) % 3u);
    ivec3 cellBase = ivec3(int(localX), int(localY + subChunkIndex * 16u), int(localZ));
    ivec3 planeBase = cellBase;
    planeBase[int(axis)] += positiveNormal ? 1 : 0;

    vec3 base = vec3(float(pc.chunkMinX + planeBase.x), float(planeBase.y), float(pc.chunkMinZ + planeBase.z));

    vec3 du = vec3(0.0);
    vec3 dv = vec3(0.0);
    du[uAxis] = float(width);
    dv[vAxis] = float(height);

    vec3 p0;
    vec3 p1;
    vec3 p2;
    vec3 p3;
    vec2 uvMax = axis == 0u ? vec2(float(height), float(width)) : vec2(float(width), float(height));

    if (axis == 0u) {
        if (positiveNormal) {
            p0 = base + dv;
            p1 = base;
            p2 = base + du;
            p3 = base + du + dv;
        } else {
            p0 = base;
            p1 = base + dv;
            p2 = base + du + dv;
            p3 = base + du;
        }
    } else if (positiveNormal) {
        p0 = base;
        p1 = base + du;
        p2 = base + du + dv;
        p3 = base + dv;
    } else {
        p0 = base + du;
        p1 = base;
        p2 = base + dv;
        p3 = base + du + dv;
    }

    int cornerIndex = GetCornerIndex(localVertexIndex, flipDiagonal, reverseWinding);
    vec3 worldPosition;
    vec2 uv;
    float ao;

    if (cornerIndex == 0) {
        worldPosition = p0;
        uv = vec2(0.0, 0.0);
        ao = DecodeAo(inPacked1);
    } else if (cornerIndex == 1) {
        worldPosition = p1;
        uv = vec2(uvMax.x, 0.0);
        ao = DecodeAo(inPacked1 >> 2u);
    } else if (cornerIndex == 2) {
        worldPosition = p2;
        uv = uvMax;
        ao = DecodeAo(inPacked1 >> 4u);
    } else {
        worldPosition = p3;
        uv = vec2(0.0, uvMax.y);
        ao = DecodeAo(inPacked1 >> 6u);
    }

    gl_Position = ubo.viewProj * vec4(worldPosition, 1.0);
    fragUv = uv;
    fragAo = ao;
}
