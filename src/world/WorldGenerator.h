#pragma once

#include "world/Block.h"

#include <cstdint>

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
};
