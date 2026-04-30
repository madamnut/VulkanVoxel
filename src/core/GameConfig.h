#pragma once

#include "world/WorldGenerator.h"

#include <string>

struct WorldConfig
{
    int chunkLoadRadius = 5;
    int chunkUploadsPerFrame = 1;
    TerrainDensityConfig terrainDensity{};
};

WorldConfig loadWorldConfigFile(
    const std::string& path,
    WorldConfig defaults,
    WorldConfig minimums,
    WorldConfig maximums);
