#pragma once

#include "world/Block.h"

#include <cstdint>
#include <shared_mutex>
#include <unordered_map>

constexpr int kTerrainBaseHeight = 192;
constexpr int kTerrainHeightRange = 40;

class WorldGenerator
{
public:
    int terrainHeightAt(int x, int z) const;
    int highestSolidYAt(int x, int z) const;
    std::uint16_t baseTerrainBlock(int y, int highestSolidY) const;
    std::uint16_t applyTerrainPostProcess(std::uint16_t blockId, int y, int highestSolidY) const;
    std::uint16_t blockIdFromColumn(int y, int highestSolidY) const;
    std::uint16_t blockIdAt(int x, int y, int z) const;
    void setBlockIdAt(int x, int y, int z, std::uint16_t blockId);

private:
    static std::int64_t blockKey(int x, int y, int z);

    mutable std::shared_mutex overrideMutex_;
    std::unordered_map<std::int64_t, std::uint16_t> blockOverrides_;
};
