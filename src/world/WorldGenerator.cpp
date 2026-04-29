#include "world/WorldGenerator.h"

#include "core/Math.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <utility>

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

float WorldGenerator::DensityGrid::valueAt(int localCellX, int cellY, int localCellZ) const
{
    return values[static_cast<std::size_t>(
        (localCellZ * countX + localCellX) * kWorldDensityVerticesY + cellY)];
}

WorldGenerator::WorldGenerator()
{
    setSeed(0);
}

void WorldGenerator::setSeed(std::uint64_t seed)
{
    seed_ = seed;
    noise_.setSeed(seed_);
    rebuildNoiseLookups();
}

std::uint64_t WorldGenerator::seed() const
{
    return seed_;
}

void WorldGenerator::setTerrainDensityConfig(const TerrainDensityConfig& config)
{
    terrainDensityConfig_ = config;
    rebuildNoiseLookups();
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
    const DensityGrid densityGrid = buildDensityGrid(x, x, z, z);
    for (int y = kChunkHeight - 1; y >= 0; --y)
    {
        if (interpolatedDensityAt(densityGrid, x, y, z) >= 0.0f)
        {
            return y + 1;
        }
    }
    return 0;
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

    if (y < 0 || y >= kChunkHeight)
    {
        return kAirBlockId;
    }

    const DensityGrid densityGrid = buildDensityGrid(x, x, z, z);
    return interpolatedDensityAt(densityGrid, x, y, z) >= 0.0f ? kRockBlockId : kAirBlockId;
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
    const int minX = chunkBaseX - GeneratedChunkColumn::kPadding;
    const int maxX = chunkBaseX + kChunkSizeX - 1 + GeneratedChunkColumn::kPadding;
    const int minZ = chunkBaseZ - GeneratedChunkColumn::kPadding;
    const int maxZ = chunkBaseZ + kChunkSizeZ - 1 + GeneratedChunkColumn::kPadding;
    const DensityGrid densityGrid = buildDensityGrid(minX, maxX, minZ, maxZ);

    for (int localPaddedZ = 0; localPaddedZ < GeneratedChunkColumn::kDepth; ++localPaddedZ)
    {
        const int z = minZ + localPaddedZ;
        for (int localPaddedX = 0; localPaddedX < GeneratedChunkColumn::kWidth; ++localPaddedX)
        {
            const int x = minX + localPaddedX;
            for (int y = 0; y < kChunkHeight; ++y)
            {
                if (interpolatedDensityAt(densityGrid, x, y, z) >= 0.0f)
                {
                    column.blockAt(localPaddedX, y, localPaddedZ) = kRockBlockId;
                }
            }
        }
    }
}

int WorldGenerator::floorDiv(int value, int divisor)
{
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

int WorldGenerator::floorMod(int value, int divisor)
{
    int result = value % divisor;
    if (result < 0)
    {
        result += divisor;
    }
    return result;
}

float WorldGenerator::lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

void WorldGenerator::rebuildNoiseLookups()
{
    const int octaveCount = std::clamp(
        terrainDensityConfig_.noise.octaves,
        0,
        kMaxTerrainDensityOctaves);
    densityOctaves_.clear();
    densityOctaves_.reserve(static_cast<std::size_t>(octaveCount));

    float frequency = static_cast<float>(terrainDensityConfig_.noise.baseFrequency);
    float amplitude = terrainDensityConfig_.noise.baseAmplitude;
    for (int octaveIndex = 0; octaveIndex < octaveCount; ++octaveIndex)
    {
        DensityOctave octave{};
        octave.frequencyXZ = std::clamp(
            static_cast<int>(std::round(frequency)),
            0,
            kWorldDensityCellsXZ / 2);
        octave.frequencyY = static_cast<float>(octave.frequencyXZ) *
            terrainDensityConfig_.noise.verticalFrequencyScale;
        octave.amplitude = amplitude;
        octave.cosX.resize(kWorldDensityCellsXZ);
        octave.sinX.resize(kWorldDensityCellsXZ);
        octave.cosZ.resize(kWorldDensityCellsXZ);
        octave.sinZ.resize(kWorldDensityCellsXZ);

        for (int i = 0; i < kWorldDensityCellsXZ; ++i)
        {
            const float angle = 2.0f * kPi *
                static_cast<float>(octave.frequencyXZ) *
                static_cast<float>(i) /
                static_cast<float>(kWorldDensityCellsXZ);
            octave.cosX[static_cast<std::size_t>(i)] = std::cos(angle);
            octave.sinX[static_cast<std::size_t>(i)] = std::sin(angle);
            octave.cosZ[static_cast<std::size_t>(i)] = octave.cosX[static_cast<std::size_t>(i)];
            octave.sinZ[static_cast<std::size_t>(i)] = octave.sinX[static_cast<std::size_t>(i)];
        }

        for (int y = 0; y < kWorldDensityVerticesY; ++y)
        {
            octave.yInput[static_cast<std::size_t>(y)] =
                static_cast<float>(y * kDensityCellSize) *
                octave.frequencyY *
                (static_cast<float>(kWorldSizeXZ) / static_cast<float>(kChunkHeight)) /
                static_cast<float>(kChunkHeight);
        }

        densityOctaves_.push_back(std::move(octave));
        frequency *= terrainDensityConfig_.noise.frequencyMultiplier;
        amplitude *= terrainDensityConfig_.noise.amplitudeMultiplier;
    }
}

float WorldGenerator::sampleDensityLattice(int cellX, int cellY, int cellZ) const
{
    const int wrappedCellX = floorMod(cellX, kWorldDensityCellsXZ);
    const int wrappedCellZ = floorMod(cellZ, kWorldDensityCellsXZ);
    const int clampedCellY = std::clamp(cellY, 0, kWorldDensityCellsY);
    const float worldY = static_cast<float>(clampedCellY * kDensityCellSize);
    const float normalizedY = worldY / static_cast<float>(kChunkHeight);
    float density = (terrainDensityConfig_.gradient.center - normalizedY) *
        terrainDensityConfig_.gradient.strength;

    for (const DensityOctave& octave : densityOctaves_)
    {
        density += noise_.sample(
            octave.cosX[static_cast<std::size_t>(wrappedCellX)],
            octave.sinX[static_cast<std::size_t>(wrappedCellX)],
            octave.cosZ[static_cast<std::size_t>(wrappedCellZ)],
            octave.sinZ[static_cast<std::size_t>(wrappedCellZ)],
            octave.yInput[static_cast<std::size_t>(clampedCellY)]) * octave.amplitude;
    }

    return density;
}

WorldGenerator::DensityGrid WorldGenerator::buildDensityGrid(
    int minBlockX,
    int maxBlockX,
    int minBlockZ,
    int maxBlockZ) const
{
    DensityGrid densityGrid{};
    densityGrid.minCellX = floorDiv(minBlockX, kDensityCellSize);
    densityGrid.minCellZ = floorDiv(minBlockZ, kDensityCellSize);
    const int maxCellX = floorDiv(maxBlockX, kDensityCellSize) + 1;
    const int maxCellZ = floorDiv(maxBlockZ, kDensityCellSize) + 1;
    densityGrid.countX = maxCellX - densityGrid.minCellX + 1;
    densityGrid.countZ = maxCellZ - densityGrid.minCellZ + 1;
    densityGrid.values.resize(static_cast<std::size_t>(
        densityGrid.countX * densityGrid.countZ * kWorldDensityVerticesY));

    for (int localZ = 0; localZ < densityGrid.countZ; ++localZ)
    {
        for (int localX = 0; localX < densityGrid.countX; ++localX)
        {
            for (int y = 0; y < kWorldDensityVerticesY; ++y)
            {
                densityGrid.values[static_cast<std::size_t>(
                    (localZ * densityGrid.countX + localX) * kWorldDensityVerticesY + y)] =
                    sampleDensityLattice(
                        densityGrid.minCellX + localX,
                        y,
                        densityGrid.minCellZ + localZ);
            }
        }
    }

    return densityGrid;
}

float WorldGenerator::interpolatedDensityAt(const DensityGrid& densityGrid, int x, int y, int z) const
{
    const int cellX = floorDiv(x, kDensityCellSize);
    const int cellY = std::clamp(y / kDensityCellSize, 0, kWorldDensityCellsY - 1);
    const int cellZ = floorDiv(z, kDensityCellSize);
    const int localCellX = cellX - densityGrid.minCellX;
    const int localCellZ = cellZ - densityGrid.minCellZ;
    const float tx = (static_cast<float>(x - cellX * kDensityCellSize) + 0.5f) /
        static_cast<float>(kDensityCellSize);
    const float ty = (static_cast<float>(y - cellY * kDensityCellSize) + 0.5f) /
        static_cast<float>(kDensityCellSize);
    const float tz = (static_cast<float>(z - cellZ * kDensityCellSize) + 0.5f) /
        static_cast<float>(kDensityCellSize);

    const float d000 = densityGrid.valueAt(localCellX, cellY, localCellZ);
    const float d100 = densityGrid.valueAt(localCellX + 1, cellY, localCellZ);
    const float d010 = densityGrid.valueAt(localCellX, cellY + 1, localCellZ);
    const float d110 = densityGrid.valueAt(localCellX + 1, cellY + 1, localCellZ);
    const float d001 = densityGrid.valueAt(localCellX, cellY, localCellZ + 1);
    const float d101 = densityGrid.valueAt(localCellX + 1, cellY, localCellZ + 1);
    const float d011 = densityGrid.valueAt(localCellX, cellY + 1, localCellZ + 1);
    const float d111 = densityGrid.valueAt(localCellX + 1, cellY + 1, localCellZ + 1);

    const float dx00 = lerp(d000, d100, tx);
    const float dx10 = lerp(d010, d110, tx);
    const float dx01 = lerp(d001, d101, tx);
    const float dx11 = lerp(d011, d111, tx);
    const float dy0 = lerp(dx00, dx10, ty);
    const float dy1 = lerp(dx01, dx11, ty);
    return lerp(dy0, dy1, tz);
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
