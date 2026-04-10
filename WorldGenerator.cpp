#include "WorldGenerator.h"

#include <algorithm>

namespace WorldGenerator {

void GenerateChunkColumn(int chunkX, int chunkZ, const TerrainConfig& terrainConfig, ChunkColumnData& outColumn) {
    (void)chunkX;
    (void)chunkZ;

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
        subChunkMesh.vertices.clear();
        subChunkMesh.indices.clear();
        subChunkMesh.dirty = true;
        subChunkMesh.revision = 0;
    }

    const int groundHeight = std::clamp(terrainConfig.flatGroundHeight, 0, kWorldSizeY);
    for (int subChunkIndex = 0; subChunkIndex < kSubChunkCountY; ++subChunkIndex) {
        const int subChunkMinY = subChunkIndex * kSubChunkSize;
        const int solidLayerCount = std::clamp(groundHeight - subChunkMinY, 0, kSubChunkSize);
        SubChunkVoxelData& subChunk = outColumn.subChunks[static_cast<std::size_t>(subChunkIndex)];

        if (solidLayerCount <= 0) {
            subChunk.isUniform = true;
            subChunk.uniformBlock = 0;
            subChunk.blocks.clear();
            continue;
        }

        if (solidLayerCount >= kSubChunkSize) {
            subChunk.isUniform = true;
            subChunk.uniformBlock = 1;
            subChunk.blocks.clear();
            continue;
        }

        subChunk.isUniform = false;
        subChunk.uniformBlock = 0;
        subChunk.blocks.assign(static_cast<std::size_t>(kSubChunkVoxelCount), 0);
        for (int localY = 0; localY < solidLayerCount; ++localY) {
            for (int localZ = 0; localZ < kSubChunkSize; ++localZ) {
                for (int localX = 0; localX < kSubChunkSize; ++localX) {
                    const int blockIndex = localY * kSubChunkSize * kSubChunkSize +
                        localZ * kSubChunkSize + localX;
                    subChunk.blocks[static_cast<std::size_t>(blockIndex)] = 1;
                }
            }
        }
    }
}

}  // namespace WorldGenerator
