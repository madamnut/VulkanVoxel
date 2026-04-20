#pragma once

#include <string>

struct TerrainConfig {
    int seed = 0;
    int wrapSizeXZ = 0;

    float baseHeight = 0.0f;
    float solidThreshold = 0.0f;
    float gradientStrength = 0.0f;
    float densityAmplitude = 0.0f;
    float featureScaleXZ = 0.0f;
    float featureScaleY = 0.0f;

    float fbmGain = 0.0f;
    int fbmOctaves = 0;
    float fbmLacunarity = 0.0f;
};

TerrainConfig LoadTerrainConfig(const std::string& path);
