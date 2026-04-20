#include "BlockRegistry.h"
#include "WorldGenerator.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace WorldGenerator {

namespace {

std::uint16_t GetTerrainBlockForDepthFromSurface(int depthFromSurface) {
    if (depthFromSurface < 0) {
        return kBlockAir;
    }
    if (depthFromSurface == 0) {
        return kBlockGrass;
    }
    if (depthFromSurface <= 3) {
        return kBlockDirt;
    }
    return kBlockRock;
}

void TryCollapseGeneratedSubChunk(SubChunkVoxelData& subChunk) {
    if (subChunk.isUniform || subChunk.blocks.empty()) {
        return;
    }

    const std::uint16_t candidate = subChunk.blocks.front();
    for (std::uint16_t blockValue : subChunk.blocks) {
        if (blockValue != candidate) {
            return;
        }
    }

    subChunk.isUniform = true;
    subChunk.uniformBlock = candidate;
    subChunk.blocks.clear();
    subChunk.blocks.shrink_to_fit();
}

}  // namespace

int ComputeTerrainHeight(int worldX, int worldZ, const TerrainConfig& terrainConfig) {
    const float waveHeight =
        std::sin(static_cast<float>(worldX) * terrainConfig.waveFrequencyX + terrainConfig.wavePhaseX) +
        std::cos(static_cast<float>(worldZ) * terrainConfig.waveFrequencyZ + terrainConfig.wavePhaseZ);
    const float heightValue = static_cast<float>(terrainConfig.seaLevel) + waveHeight * terrainConfig.waveAmplitude;
    return std::clamp(static_cast<int>(std::lround(heightValue)), 0, kWorldSizeY);
}

std::uint16_t SampleBlock(int worldX, int worldY, int worldZ, const TerrainConfig& terrainConfig) {
    if (worldY < 0 || worldY >= kWorldSizeY) {
        return kBlockAir;
    }

    const int terrainHeight = ComputeTerrainHeight(worldX, worldZ, terrainConfig);
    if (worldY >= terrainHeight) {
        return kBlockAir;
    }

    const int depthFromSurface = terrainHeight - 1 - worldY;
    return GetTerrainBlockForDepthFromSurface(depthFromSurface);
}

void GenerateChunkColumn(int chunkX, int chunkZ, const TerrainConfig& terrainConfig, ChunkColumnData& outColumn) {
    outColumn.subChunks.clear();
    outColumn.subChunks.resize(kSubChunkCountY);
    outColumn.subChunkMeshes.clear();
    outColumn.subChunkMeshes.resize(kSubChunkCountY);
    outColumn.modified = false;

    for (SubChunkVoxelData& subChunk : outColumn.subChunks) {
        subChunk.blocks.clear();
        subChunk.uniformBlock = 0;
        subChunk.isUniform = true;
    }

    for (SubChunkMeshData& subChunkMesh : outColumn.subChunkMeshes) {
        subChunkMesh.quads.clear();
        subChunkMesh.dirty = true;
        subChunkMesh.revision = 0;
    }

    std::array<int, kChunkSizeX * kChunkSizeZ> columnHeights{};
    int minHeight = kWorldSizeY;
    int maxHeight = 0;
        for (int localZ = 0; localZ < kChunkSizeZ; ++localZ) {
            for (int localX = 0; localX < kChunkSizeX; ++localX) {
                const int worldX = chunkX * kChunkSizeX + localX;
                const int worldZ = chunkZ * kChunkSizeZ + localZ;
                const int groundHeight = ComputeTerrainHeight(worldX, worldZ, terrainConfig);
                columnHeights[static_cast<std::size_t>(localZ * kChunkSizeX + localX)] = groundHeight;
                minHeight = std::min(minHeight, groundHeight);
                maxHeight = std::max(maxHeight, groundHeight);
            }
        }

    for (int subChunkIndex = 0; subChunkIndex < kSubChunkCountY; ++subChunkIndex) {
        const int subChunkMinY = subChunkIndex * kSubChunkSize;
        SubChunkVoxelData& subChunk = outColumn.subChunks[static_cast<std::size_t>(subChunkIndex)];
        const int subChunkMaxY = subChunkMinY + kSubChunkSize;

        if (maxHeight <= subChunkMinY) {
            subChunk.isUniform = true;
            subChunk.uniformBlock = kBlockAir;
            subChunk.blocks.clear();
            continue;
        }

        if (subChunkMaxY <= (minHeight - 4)) {
            subChunk.isUniform = true;
            subChunk.uniformBlock = kBlockRock;
            subChunk.blocks.clear();
            continue;
        }

        subChunk.isUniform = false;
        subChunk.uniformBlock = kBlockAir;
        subChunk.blocks.assign(static_cast<std::size_t>(kSubChunkVoxelCount), 0);
        for (int localZ = 0; localZ < kChunkSizeZ; ++localZ) {
            for (int localX = 0; localX < kChunkSizeX; ++localX) {
                const int groundHeight = columnHeights[static_cast<std::size_t>(localZ * kChunkSizeX + localX)];
                const int localSurfaceY = groundHeight - 1 - subChunkMinY;
                const int solidLayerCount = std::clamp(groundHeight - subChunkMinY, 0, kSubChunkSize);
                for (int localY = 0; localY < solidLayerCount; ++localY) {
                    const int depthFromSurface = localSurfaceY - localY;
                    const int blockIndex = localY * kSubChunkSize * kSubChunkSize +
                        localZ * kSubChunkSize + localX;
                    subChunk.blocks[static_cast<std::size_t>(blockIndex)] =
                        GetTerrainBlockForDepthFromSurface(depthFromSurface);
                }
            }
        }

        TryCollapseGeneratedSubChunk(subChunk);
    }
}

}  // namespace WorldGenerator
