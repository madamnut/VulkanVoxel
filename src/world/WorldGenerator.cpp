#include "world/WorldGenerator.h"

#include <algorithm>
#include <cmath>

int WorldGenerator::terrainHeightAt(int x, int z) const
{
    constexpr float amplitude = static_cast<float>(kTerrainHeightRange) * 0.5f;
    const float xWave = std::sin(static_cast<float>(x) * 0.035f);
    const float zWave = std::cos(static_cast<float>(z) * 0.041f);
    const float diagonalWave = std::sin(static_cast<float>(x + z) * 0.018f) * 0.35f;
    const float normalizedWave = std::clamp((xWave + zWave + diagonalWave) / 2.35f, -1.0f, 1.0f);
    return kTerrainBaseHeight + static_cast<int>(std::lround(normalizedWave * amplitude));
}

int WorldGenerator::highestSolidYAt(int x, int z) const
{
    const int terrainHeight = terrainHeightAt(x, z);
    return terrainHeight > 0 ? terrainHeight - 1 : -1;
}

std::uint16_t WorldGenerator::baseTerrainBlock(int y, int highestSolidY) const
{
    if (y < 0 || highestSolidY < 0 || y > highestSolidY)
    {
        return kAirBlockId;
    }
    return kRockBlockId;
}

std::uint16_t WorldGenerator::applyTerrainPostProcess(std::uint16_t blockId, int y, int highestSolidY) const
{
    if (blockId == kAirBlockId)
    {
        return blockId;
    }
    if (y == 0)
    {
        return kBedrockBlockId;
    }
    if (y == highestSolidY)
    {
        return kGrassBlockId;
    }
    if (y >= highestSolidY - 4 && y < highestSolidY)
    {
        return kDirtBlockId;
    }
    return blockId;
}

std::uint16_t WorldGenerator::blockIdFromColumn(int y, int highestSolidY) const
{
    return applyTerrainPostProcess(baseTerrainBlock(y, highestSolidY), y, highestSolidY);
}

std::uint16_t WorldGenerator::blockIdAt(int x, int y, int z) const
{
    return blockIdFromColumn(y, highestSolidYAt(x, z));
}
