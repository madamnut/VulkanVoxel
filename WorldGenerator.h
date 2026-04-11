#pragma once

#include "TerrainConfig.h"
#include "VoxelWorld.h"

namespace WorldGenerator {

int ComputeTerrainHeight(int worldX, int worldZ, const TerrainConfig& terrainConfig);
std::uint16_t SampleBlock(int worldX, int worldY, int worldZ, const TerrainConfig& terrainConfig);
void GenerateChunkColumn(int chunkX, int chunkZ, const TerrainConfig& terrainConfig, ChunkColumnData& outColumn);

}  // namespace WorldGenerator
