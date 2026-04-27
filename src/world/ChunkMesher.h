#pragma once

#include "world/BlockRegistry.h"
#include "world/ChunkTypes.h"
#include "world/WorldGenerator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

class ChunkMesher
{
public:
    ChunkMesher(
        const WorldGenerator& worldGenerator,
        const BlockRegistry& blockRegistry);

    SubchunkBuildResult buildSubchunkMesh(ChunkBuildRequest request) const;

private:
    const BlockDefinition* blockDefinitionForId(std::uint16_t blockId) const;
    bool isSolidBlock(std::uint16_t blockId) const;
    std::uint32_t textureLayerForBlockFace(std::uint16_t blockId, BlockFace face) const;

    void appendSubchunkMesh(
        int chunkX,
        int chunkZ,
        int subchunkY,
        const std::array<std::vector<BlockVertex>, kSubchunkSize>& verticesByLocalY,
        const std::array<std::vector<std::uint32_t>, kSubchunkSize>& indicesByLocalY,
        std::vector<BlockVertex>& chunkVertices,
        std::vector<std::uint32_t>& chunkIndices,
        std::vector<SubchunkDraw>& subchunkDraws) const;

    const WorldGenerator& worldGenerator_;
    const BlockRegistry& blockRegistry_;
};
