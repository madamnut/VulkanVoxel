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
        return buildChunkMeshFromPreparedColumn(coord, generation, column);
    }

    ChunkBuildResult buildChunkMeshFromPreparedColumn(
        ChunkCoord coord,
        std::uint64_t generation,
        const GeneratedChunkColumn& column) const;

    void appendChunkMeshFromPreparedColumn(
        ChunkBuildResult& result,
        ChunkCoord coord,
        std::uint64_t generation,
        const GeneratedChunkColumn& column) const;
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

    const WorldGenerator& worldGenerator_;
    const BlockRegistry& blockRegistry_;
    std::unique_ptr<CachedBlockMeshInfo[]> blockMeshInfo_;
    std::uint32_t waterTextureLayer_ = 0;
};
