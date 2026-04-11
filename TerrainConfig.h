#pragma once

#include <string>

struct TerrainConfig {
    int seed = 0;

    int seaLevel = 0;
    int flatGroundHeight = 0;
    float solidThreshold = 0.0f;
    float waveAmplitude = 0.0f;
    float waveFrequencyX = 0.0f;
    float waveFrequencyZ = 0.0f;
    float wavePhaseX = 0.0f;
    float wavePhaseZ = 0.0f;

    float noiseFeatureScale = 0.0f;
    float noiseOutputMin = 0.0f;
    float noiseOutputMax = 0.0f;

    float fbmGain = 0.0f;
    float fbmWeightedStrength = 0.0f;
    int fbmOctaves = 0;
    float fbmLacunarity = 0.0f;

    bool warpEnabled = false;
    float warpAmplitude = 0.0f;
    float warpFeatureScale = 0.0f;
    int warpSeedOffset = 0;
    float warpXAmplitudeScaling = 0.0f;
    float warpYAmplitudeScaling = 0.0f;
    float warpZAmplitudeScaling = 0.0f;
    float warpWAmplitudeScaling = 0.0f;

    float gradientCenterY = 0.0f;
    float gradientScale = 0.0f;
    float gradientOffset = 0.0f;
};

TerrainConfig LoadTerrainConfig(const std::string& path);
