#pragma once

#include "world/BlockRegistry.h"
#include "world/ChunkTypes.h"
#include "world/WorldGenerator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class ChunkMesher
{
public:
    ChunkMesher(
        const WorldGenerator& worldGenerator,
        const BlockRegistry& blockRegistry);

    void setWaterTextureLayer(std::uint32_t textureLayer);
    ChunkBuildResult buildChunkMesh(ChunkCoord coord, std::uint64_t generation) const;
    ChunkBuildResult buildChunkMeshFromColumn(
        ChunkCoord coord,
        std::uint64_t generation,
        const GeneratedChunkColumn& column) const
    {
        ChunkBuildResult result{};
        result.coord = coord;
        result.subchunks.reserve(kSubchunksPerChunk);
        result.fluidSubchunks.reserve(kSubchunksPerChunk);

        for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
        {
            SubchunkBuildResult subchunk = buildSubchunkMesh(
                {
                    coord,
                    subchunkY,
                    0,
                    generation,
                },
                column);
            if (!subchunk.indices.empty())
            {
                SubchunkDraw draw{};
                draw.chunkX = coord.x;
                draw.chunkZ = coord.z;
                draw.subchunkY = subchunkY;
                draw.range.vertexCount = static_cast<std::uint32_t>(subchunk.vertices.size());
                draw.range.firstIndex = static_cast<std::uint32_t>(result.indices.size());
                draw.range.indexCount = static_cast<std::uint32_t>(subchunk.indices.size());
                draw.range.vertexOffset = static_cast<std::int32_t>(result.vertices.size());

                result.vertices.insert(result.vertices.end(), subchunk.vertices.begin(), subchunk.vertices.end());
                result.indices.insert(result.indices.end(), subchunk.indices.begin(), subchunk.indices.end());
                result.subchunks.push_back(draw);
            }

            if (!subchunk.fluidIndices.empty())
            {
                SubchunkDraw draw{};
                draw.chunkX = coord.x;
                draw.chunkZ = coord.z;
                draw.subchunkY = subchunkY;
                draw.range.vertexCount = static_cast<std::uint32_t>(subchunk.fluidVertices.size());
                draw.range.firstIndex = static_cast<std::uint32_t>(result.fluidIndices.size());
                draw.range.indexCount = static_cast<std::uint32_t>(subchunk.fluidIndices.size());
                draw.range.vertexOffset = static_cast<std::int32_t>(result.fluidVertices.size());

                result.fluidVertices.insert(
                    result.fluidVertices.end(),
                    subchunk.fluidVertices.begin(),
                    subchunk.fluidVertices.end());
                result.fluidIndices.insert(
                    result.fluidIndices.end(),
                    subchunk.fluidIndices.begin(),
                    subchunk.fluidIndices.end());
                result.fluidSubchunks.push_back(draw);
            }
        }

        return result;
    }
    SubchunkBuildResult buildSubchunkMesh(ChunkBuildRequest request) const;

private:
    struct CachedBlockMeshInfo
    {
        BlockRenderShape renderShape = BlockRenderShape::None;
        bool collision = false;
        bool faceOccluder = false;
        bool aoOccluder = false;
        std::array<std::uint32_t, 3> textureLayers{};
    };

    void refreshBlockMeshInfo();
    const CachedBlockMeshInfo& blockInfoForId(std::uint16_t blockId) const;
    bool isCollisionBlock(std::uint16_t blockId) const;
    bool isCubeBlock(std::uint16_t blockId) const;
    bool isFaceOccluderBlock(std::uint16_t blockId) const;
    bool isAoOccluderBlock(std::uint16_t blockId) const;
    std::uint32_t textureLayerForBlockFace(std::uint16_t blockId, BlockFace face) const;
    ChunkBuildResult buildChunkMesh(
        ChunkCoord coord,
        std::uint64_t generation,
        const GeneratedChunkColumn& column) const;
    SubchunkBuildResult buildSubchunkMesh(
        ChunkBuildRequest request,
        const GeneratedChunkColumn& column) const;

    void appendSubchunkMesh(
        const std::array<std::vector<BlockVertex>, kSubchunkSize>& verticesByLocalY,
        const std::array<std::vector<std::uint32_t>, kSubchunkSize>& indicesByLocalY,
        std::vector<BlockVertex>& chunkVertices,
        std::vector<std::uint32_t>& chunkIndices) const;

    const WorldGenerator& worldGenerator_;
    const BlockRegistry& blockRegistry_;
    std::unique_ptr<CachedBlockMeshInfo[]> blockMeshInfo_;
    std::uint32_t waterTextureLayer_ = 0;
};
