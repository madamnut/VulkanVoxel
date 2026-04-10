#pragma once

#include "TerrainConfig.h"
#include "VoxelWorld.h"

namespace WorldGenerator {

void GenerateChunkColumn(int chunkX, int chunkZ, const TerrainConfig& terrainConfig, ChunkColumnData& outColumn);

}  // namespace WorldGenerator
