#pragma once

namespace PeriodicSimplex {

struct FbmSettings {
    int seed = 1337;
    int wrapSizeXZ = 65536;
    float featureScaleXZ = 128.0f;
    float featureScaleY = 96.0f;
    int octaves = 4;
    float gain = 0.5f;
    float lacunarity = 2.0f;
};

float SampleTileableXZ3DFbm(float worldX, float worldY, float worldZ, const FbmSettings& settings);

}  // namespace PeriodicSimplex
