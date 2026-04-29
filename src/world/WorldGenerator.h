#pragma once

#include "world/Block.h"
#include "world/ChunkTypes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

constexpr int kTerrainBaseHeight = 192;
constexpr int kTerrainHeightRange = 40;
constexpr int kTerrainDirtDepth = 4;

struct GeneratedChunkColumn
{
    static constexpr int kPadding = 1;
    static constexpr int kWidth = kChunkSizeX + kPadding * 2;
    static constexpr int kDepth = kChunkSizeZ + kPadding * 2;
    static constexpr int kHeight = kChunkHeight;

    std::array<std::uint16_t, kWidth * kHeight * kDepth> blockIds{};

    static std::size_t index(int localPaddedX, int y, int localPaddedZ);
    std::uint16_t blockAt(int localPaddedX, int y, int localPaddedZ) const;
    std::uint16_t& blockAt(int localPaddedX, int y, int localPaddedZ);
};

class WorldGenerator
{
public:
    int terrainHeightAt(int x, int z) const;
    int highestSolidYAt(int x, int z) const;
    std::uint16_t baseTerrainBlock(int y, int highestSolidY) const;
    std::uint16_t applyTerrainPostProcess(std::uint16_t blockId, int y, int highestSolidY) const;
    std::uint16_t blockIdFromColumn(int y, int highestSolidY) const;
    std::uint16_t blockIdAt(int x, int y, int z) const;
    GeneratedChunkColumn generateChunkColumn(ChunkCoord coord) const;
    GeneratedChunkColumn generateChunkColumn(ChunkCoord coord, const std::vector<std::uint16_t>& blockIds) const;
    std::vector<std::uint16_t> generateChunkBlocks(ChunkCoord coord) const;
    void setBlockIdAt(int x, int y, int z, std::uint16_t blockId);

private:
    struct BlockOverride
    {
        int x = 0;
        int y = 0;
        int z = 0;
        std::uint16_t blockId = kAirBlockId;
    };

    static std::int64_t blockKey(int x, int y, int z);
    void generateBaseTerrain(ChunkCoord coord, GeneratedChunkColumn& column) const;
    void applySurfaceMaterials(GeneratedChunkColumn& column) const;
    void applyBlockOverrides(ChunkCoord coord, GeneratedChunkColumn& column) const;

    mutable std::shared_mutex overrideMutex_;
    std::unordered_map<std::int64_t, BlockOverride> blockOverrides_;
};
