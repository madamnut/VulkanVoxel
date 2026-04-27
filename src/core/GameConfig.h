#pragma once

#include <string>

struct WorldConfig
{
    int chunkLoadRadius = 5;
    int chunkUploadsPerFrame = 1;
    int chunkBuildThreads = 4;
};

WorldConfig loadWorldConfigFile(
    const std::string& path,
    WorldConfig defaults,
    WorldConfig minimums,
    WorldConfig maximums);
