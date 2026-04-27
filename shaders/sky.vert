#version 450

layout(set = 0, binding = 0) uniform SkyUniform
{
    vec4 camera; // yaw, pitch, aspect, tan(fovY / 2)
    vec4 sunDirection;
    vec4 moonDirection;
    vec4 spriteScale; // sun height in NDC, moon height in NDC
} ubo;

layout(location = 0) out vec2 outUv;
layout(location = 1) flat out int outSpriteIndex;

vec3 cameraForward(float yaw, float pitch)
{
    return normalize(vec3(cos(pitch) * cos(yaw), sin(pitch), cos(pitch) * sin(yaw)));
}

void main()
{
    const vec2 offsets[6] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0)
    );

    int spriteIndex = gl_InstanceIndex;
    vec3 worldDirection = normalize(spriteIndex == 0 ? ubo.sunDirection.xyz : ubo.moonDirection.xyz);

    float yaw = ubo.camera.x;
    float pitch = ubo.camera.y;
    float aspect = ubo.camera.z;
    float tanHalfFovY = ubo.camera.w;

    vec3 forward = cameraForward(yaw, pitch);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));

    vec3 viewDirection = vec3(
        dot(worldDirection, right),
        dot(worldDirection, up),
        dot(worldDirection, forward)
    );

    vec2 offscreen = vec2(3.0, 3.0);
    vec2 center = offscreen;
    if (viewDirection.z > 0.001)
    {
        center = vec2(
            (viewDirection.x / viewDirection.z) / (tanHalfFovY * aspect),
            (viewDirection.y / viewDirection.z) / tanHalfFovY
        );
    }

    vec2 corner = offsets[gl_VertexIndex];
    float spriteHeight = spriteIndex == 0 ? ubo.spriteScale.x : ubo.spriteScale.y;
    vec2 spriteHalfSize = vec2(spriteHeight / aspect, spriteHeight);

    outUv = corner * 0.5 + vec2(0.5);
    outUv.y = 1.0 - outUv.y;
    outSpriteIndex = spriteIndex;
    gl_Position = vec4(center + corner * spriteHalfSize, 0.0, 1.0);
}

