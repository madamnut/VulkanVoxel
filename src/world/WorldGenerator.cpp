#include "world/WorldGenerator.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

std::size_t GeneratedChunkColumn::index(int localPaddedX, int y, int localPaddedZ)
{
    return static_cast<std::size_t>((y * kDepth + localPaddedZ) * kWidth + localPaddedX);
}

std::uint16_t GeneratedChunkColumn::blockAt(int localPaddedX, int y, int localPaddedZ) const
{
    if (localPaddedX < 0 || localPaddedX >= kWidth ||
        y < 0 || y >= kHeight ||
        localPaddedZ < 0 || localPaddedZ >= kDepth)
    {
        return kAirBlockId;
    }
    return blockIds[index(localPaddedX, y, localPaddedZ)];
}

std::uint16_t& GeneratedChunkColumn::blockAt(int localPaddedX, int y, int localPaddedZ)
{
    return blockIds[index(localPaddedX, y, localPaddedZ)];
}

std::int64_t WorldGenerator::blockKey(int x, int y, int z)
{
    constexpr std::int64_t mask = (1ll << 21) - 1ll;
    return ((static_cast<std::int64_t>(x) & mask) << 43) |
           ((static_cast<std::int64_t>(z) & mask) << 22) |
           (static_cast<std::int64_t>(y) & ((1ll << 22) - 1ll));
}

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
    {
        std::shared_lock lock(overrideMutex_);
        const auto overrideIt = blockOverrides_.find(blockKey(x, y, z));
        if (overrideIt != blockOverrides_.end())
        {
            return overrideIt->second.blockId;
        }
    }

    return blockIdFromColumn(y, highestSolidYAt(x, z));
}

GeneratedChunkColumn WorldGenerator::generateChunkColumn(ChunkCoord coord) const
{
    GeneratedChunkColumn column{};
    generateBaseTerrain(coord, column);
    applySurfaceMaterials(column);
    applyBlockOverrides(coord, column);
    return column;
}

GeneratedChunkColumn WorldGenerator::generateChunkColumn(
    ChunkCoord coord,
    const std::vector<std::uint16_t>& blockIds) const
{
    if (blockIds.size() != kChunkBlockCount)
    {
        throw std::runtime_error("Cannot build chunk column from an invalid block count.");
    }

    GeneratedChunkColumn column = generateChunkColumn(coord);
    for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
    {
        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int y = 0; y < kChunkHeight; ++y)
            {
                column.blockAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding) =
                    blockIds[chunkBlockIndex(localX, y, localZ)];
            }
        }
    }
    return column;
}

std::vector<std::uint16_t> WorldGenerator::generateChunkBlocks(ChunkCoord coord) const
{
    const GeneratedChunkColumn column = generateChunkColumn(coord);
    std::vector<std::uint16_t> blockIds(kChunkBlockCount);
    for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
    {
        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int y = 0; y < kChunkHeight; ++y)
            {
                blockIds[chunkBlockIndex(localX, y, localZ)] =
                    column.blockAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding);
            }
        }
    }
    return blockIds;
}

void WorldGenerator::generateBaseTerrain(ChunkCoord coord, GeneratedChunkColumn& column) const
{
    const int chunkBaseX = coord.x * kChunkSizeX;
    const int chunkBaseZ = coord.z * kChunkSizeZ;

    for (int localPaddedZ = 0; localPaddedZ < GeneratedChunkColumn::kDepth; ++localPaddedZ)
    {
        const int z = chunkBaseZ + localPaddedZ - GeneratedChunkColumn::kPadding;
        for (int localPaddedX = 0; localPaddedX < GeneratedChunkColumn::kWidth; ++localPaddedX)
        {
            const int x = chunkBaseX + localPaddedX - GeneratedChunkColumn::kPadding;
            const int highestSolidY = highestSolidYAt(x, z);
            const int clampedHighestSolidY = std::clamp(highestSolidY, -1, kChunkHeight - 1);
            for (int y = 0; y <= clampedHighestSolidY; ++y)
            {
                column.blockAt(localPaddedX, y, localPaddedZ) = kRockBlockId;
            }
        }
    }
}

void WorldGenerator::applySurfaceMaterials(GeneratedChunkColumn& column) const
{
    for (int localPaddedZ = 0; localPaddedZ < GeneratedChunkColumn::kDepth; ++localPaddedZ)
    {
        for (int localPaddedX = 0; localPaddedX < GeneratedChunkColumn::kWidth; ++localPaddedX)
        {
            bool foundHighestSolid = false;
            int dirtRemaining = kTerrainDirtDepth;
            for (int y = kChunkHeight - 1; y >= 0; --y)
            {
                std::uint16_t& blockId = column.blockAt(localPaddedX, y, localPaddedZ);
                if (blockId == kAirBlockId)
                {
                    continue;
                }

                if (!foundHighestSolid)
                {
                    blockId = kGrassBlockId;
                    foundHighestSolid = true;
                    continue;
                }

                if (dirtRemaining > 0)
                {
                    blockId = kDirtBlockId;
                    --dirtRemaining;
                }
            }

            std::uint16_t& bottomBlockId = column.blockAt(localPaddedX, 0, localPaddedZ);
            if (bottomBlockId != kAirBlockId)
            {
                bottomBlockId = kBedrockBlockId;
            }
        }
    }
}

void WorldGenerator::applyBlockOverrides(ChunkCoord coord, GeneratedChunkColumn& column) const
{
    const int minX = coord.x * kChunkSizeX - GeneratedChunkColumn::kPadding;
    const int maxX = coord.x * kChunkSizeX + kChunkSizeX - 1 + GeneratedChunkColumn::kPadding;
    const int minZ = coord.z * kChunkSizeZ - GeneratedChunkColumn::kPadding;
    const int maxZ = coord.z * kChunkSizeZ + kChunkSizeZ - 1 + GeneratedChunkColumn::kPadding;

    std::shared_lock lock(overrideMutex_);
    for (const auto& entry : blockOverrides_)
    {
        const BlockOverride& blockOverride = entry.second;
        if (blockOverride.x < minX || blockOverride.x > maxX ||
            blockOverride.y < 0 || blockOverride.y >= kChunkHeight ||
            blockOverride.z < minZ || blockOverride.z > maxZ)
        {
            continue;
        }

        const int localPaddedX = blockOverride.x - minX;
        const int localPaddedZ = blockOverride.z - minZ;
        column.blockAt(localPaddedX, blockOverride.y, localPaddedZ) = blockOverride.blockId;
    }
}

void WorldGenerator::setBlockIdAt(int x, int y, int z, std::uint16_t blockId)
{
    const std::uint16_t generatedBlockId = blockIdFromColumn(y, highestSolidYAt(x, z));
    std::unique_lock lock(overrideMutex_);
    const std::int64_t key = blockKey(x, y, z);
    if (blockId == generatedBlockId)
    {
        blockOverrides_.erase(key);
        return;
    }

    blockOverrides_[key] = {x, y, z, blockId};
}
