#include "world/WorldGenerator.h"

#include "core/Math.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace
{
constexpr std::uint32_t kLandformCurvePairCount = 1000;
constexpr int kPlantSpawnChancePercent = 40;
constexpr int kTreeDecorationReach = 2;
constexpr int kTreeSpawnChancePercent = 2;
constexpr int kTreeMinTrunkHeight = 4;
constexpr int kTreeTrunkHeightRange = 3;

std::size_t checkedColumnIndex(
    int localPaddedX,
    int y,
    int localPaddedZ,
    std::size_t size)
{
    if (localPaddedX < 0 || localPaddedX >= GeneratedChunkColumn::kWidth ||
        y < 0 || y >= GeneratedChunkColumn::kHeight ||
        localPaddedZ < 0 || localPaddedZ >= GeneratedChunkColumn::kDepth)
    {
        throw std::out_of_range("GeneratedChunkColumn coordinate is out of range.");
    }

    const std::size_t index = GeneratedChunkColumn::index(localPaddedX, y, localPaddedZ);
    if (index >= size)
    {
        throw std::out_of_range("GeneratedChunkColumn storage is invalid.");
    }
    return index;
}
}

GeneratedChunkColumn::GeneratedChunkColumn()
    : blockIds(kCellCount, kAirBlockId)
    , fluidIds(kCellCount, kNoFluidId)
    , fluidAmounts(kCellCount, 0)
{
}

std::size_t GeneratedChunkColumn::index(int localPaddedX, int y, int localPaddedZ)
{
    return static_cast<std::size_t>((y * kDepth + localPaddedZ) * kWidth + localPaddedX);
}

std::uint16_t GeneratedChunkColumn::blockAt(int localPaddedX, int y, int localPaddedZ) const
{
    if (localPaddedX < 0 || localPaddedX >= kWidth ||
        y < 0 || y >= kHeight ||
        localPaddedZ < 0 || localPaddedZ >= kDepth ||
        blockIds.size() != kCellCount)
    {
        return kAirBlockId;
    }
    return blockIds[index(localPaddedX, y, localPaddedZ)];
}

std::uint16_t& GeneratedChunkColumn::blockAt(int localPaddedX, int y, int localPaddedZ)
{
    return blockIds[checkedColumnIndex(localPaddedX, y, localPaddedZ, blockIds.size())];
}

std::uint8_t GeneratedChunkColumn::fluidIdAt(int localPaddedX, int y, int localPaddedZ) const
{
    if (localPaddedX < 0 || localPaddedX >= kWidth ||
        y < 0 || y >= kHeight ||
        localPaddedZ < 0 || localPaddedZ >= kDepth ||
        fluidIds.size() != kCellCount)
    {
        return kNoFluidId;
    }
    return fluidIds[index(localPaddedX, y, localPaddedZ)];
}

std::uint8_t& GeneratedChunkColumn::fluidIdAt(int localPaddedX, int y, int localPaddedZ)
{
    return fluidIds[checkedColumnIndex(localPaddedX, y, localPaddedZ, fluidIds.size())];
}

std::uint8_t GeneratedChunkColumn::fluidAt(int localPaddedX, int y, int localPaddedZ) const
{
    if (localPaddedX < 0 || localPaddedX >= kWidth ||
        y < 0 || y >= kHeight ||
        localPaddedZ < 0 || localPaddedZ >= kDepth ||
        fluidIds.size() != kCellCount ||
        fluidAmounts.size() != kCellCount)
    {
        return 0;
    }
    if (fluidIds[index(localPaddedX, y, localPaddedZ)] == kNoFluidId)
    {
        return 0;
    }
    return fluidAmounts[index(localPaddedX, y, localPaddedZ)];
}

std::uint8_t& GeneratedChunkColumn::fluidAt(int localPaddedX, int y, int localPaddedZ)
{
    return fluidAmounts[checkedColumnIndex(localPaddedX, y, localPaddedZ, fluidAmounts.size())];
}

float WorldGenerator::DensityGrid::valueAt(int localCellX, int cellY, int localCellZ) const
{
    if (localCellX < 0 || localCellX >= countX ||
        localCellZ < 0 || localCellZ >= countZ ||
        cellY < 0 || cellY >= kWorldDensityVerticesY)
    {
        throw std::out_of_range("DensityGrid coordinate is out of range.");
    }

    const std::size_t index = static_cast<std::size_t>(
        (localCellZ * countX + localCellX) * kWorldDensityVerticesY + cellY);
    if (index >= values.size())
    {
        throw std::out_of_range("DensityGrid storage is invalid.");
    }
    return values[index];
}

WorldGenerator::WorldGenerator()
{
    setSeed(0);
}

void WorldGenerator::setSeed(std::uint64_t seed)
{
    std::unique_lock lock(generatorMutex_);
    seed_ = seed;
    noise_.setSeed(seed_);
    landformNoise_.setSeed(seed_ ^ 0x6a09e667f3bcc909ull);
    rebuildNoiseLookups();
}

std::uint64_t WorldGenerator::seed() const
{
    return seed_;
}

void WorldGenerator::setTerrainDensityConfig(const TerrainDensityConfig& config)
{
    std::unique_lock lock(generatorMutex_);
    terrainDensityConfig_ = config;
    rebuildNoiseLookups();
}

bool WorldGenerator::loadLandformCurveFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::unique_lock lock(generatorMutex_);
        landformLookup_.rawSamples.clear();
        landformLookup_.centerOffsetSamples.clear();
        return false;
    }

    std::uint32_t pairCount = 0;
    file.read(reinterpret_cast<char*>(&pairCount), sizeof(pairCount));
    if (!file || pairCount != kLandformCurvePairCount)
    {
        std::unique_lock lock(generatorMutex_);
        landformLookup_.rawSamples.clear();
        landformLookup_.centerOffsetSamples.clear();
        return false;
    }

    std::vector<float> rawSamples(pairCount);
    std::vector<float> centerOffsetSamples(pairCount);
    for (std::uint32_t i = 0; i < pairCount; ++i)
    {
        float raw = 0.0f;
        float centerOffset = 0.0f;
        file.read(reinterpret_cast<char*>(&raw), sizeof(raw));
        file.read(reinterpret_cast<char*>(&centerOffset), sizeof(centerOffset));
        if (!file || (i > 0 && raw <= rawSamples[static_cast<std::size_t>(i - 1)]))
        {
            std::unique_lock lock(generatorMutex_);
            landformLookup_.rawSamples.clear();
            landformLookup_.centerOffsetSamples.clear();
            return false;
        }

        rawSamples[static_cast<std::size_t>(i)] = raw;
        centerOffsetSamples[static_cast<std::size_t>(i)] = centerOffset;
    }

    if (rawSamples.front() > -1.0f || rawSamples.back() < 1.0f)
    {
        std::unique_lock lock(generatorMutex_);
        landformLookup_.rawSamples.clear();
        landformLookup_.centerOffsetSamples.clear();
        return false;
    }

    std::unique_lock lock(generatorMutex_);
    landformLookup_.rawSamples = std::move(rawSamples);
    landformLookup_.centerOffsetSamples = std::move(centerOffsetSamples);
    return true;
}

std::int64_t WorldGenerator::blockKey(int x, int y, int z)
{
    constexpr std::int64_t mask = (1ll << 21) - 1ll;
    return ((static_cast<std::int64_t>(x) & mask) << 43) |
           ((static_cast<std::int64_t>(z) & mask) << 22) |
           (static_cast<std::int64_t>(y) & ((1ll << 22) - 1ll));
}

std::uint64_t WorldGenerator::mixHash(std::uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

std::uint64_t WorldGenerator::treeHash(std::uint64_t seed, int x, int z, std::uint64_t salt)
{
    std::uint64_t value = seed ^ salt;
    value ^= mixHash(static_cast<std::uint32_t>(x));
    value ^= mixHash(static_cast<std::uint32_t>(z)) + 0x9e3779b97f4a7c15ull + (value << 6) + (value >> 2);
    return mixHash(value);
}

int WorldGenerator::terrainHeightAt(int x, int z) const
{
    std::shared_lock lock(generatorMutex_);
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

    std::shared_lock lock(generatorMutex_);
    const DensityGrid densityGrid = buildDensityGrid(x, x, z, z);
    return interpolatedDensityAt(densityGrid, x, y, z) >= 0.0f ? kRockBlockId : kAirBlockId;
}

float WorldGenerator::landformCenterOffsetAt(int x, int z) const
{
    std::shared_lock lock(generatorMutex_);
    const int cellX = floorDiv(x, kDensityCellSize);
    const int cellZ = floorDiv(z, kDensityCellSize);
    const float tx = (static_cast<float>(x - cellX * kDensityCellSize) + 0.5f) /
        static_cast<float>(kDensityCellSize);
    const float tz = (static_cast<float>(z - cellZ * kDensityCellSize) + 0.5f) /
        static_cast<float>(kDensityCellSize);

    const float b00 = sampleLandformCenterOffset(cellX, cellZ);
    const float b10 = sampleLandformCenterOffset(cellX + 1, cellZ);
    const float b01 = sampleLandformCenterOffset(cellX, cellZ + 1);
    const float b11 = sampleLandformCenterOffset(cellX + 1, cellZ + 1);
    return lerp(lerp(b00, b10, tx), lerp(b01, b11, tx), tz);
}

float WorldGenerator::landformRawAt(int x, int z) const
{
    std::shared_lock lock(generatorMutex_);
    const int cellX = floorDiv(x, kDensityCellSize);
    const int cellZ = floorDiv(z, kDensityCellSize);
    const float tx = (static_cast<float>(x - cellX * kDensityCellSize) + 0.5f) /
        static_cast<float>(kDensityCellSize);
    const float tz = (static_cast<float>(z - cellZ * kDensityCellSize) + 0.5f) /
        static_cast<float>(kDensityCellSize);

    const float r00 = sampleLandformRaw(cellX, cellZ);
    const float r10 = sampleLandformRaw(cellX + 1, cellZ);
    const float r01 = sampleLandformRaw(cellX, cellZ + 1);
    const float r11 = sampleLandformRaw(cellX + 1, cellZ + 1);
    return lerp(lerp(r00, r10, tx), lerp(r01, r11, tx), tz);
}

GeneratedChunkColumn WorldGenerator::generateChunkColumn(ChunkCoord coord) const
{
    std::shared_lock lock(generatorMutex_);
    GeneratedChunkColumn column{};
    generateBaseTerrain(coord, column);
    applySurfaceMaterials(column);
    applyPlantDecorations(coord, column);
    applyTreeDecorations(coord, column);
    applyBlockOverrides(coord, column);
    return column;
}

GeneratedChunkColumn WorldGenerator::generateChunkColumn(
    ChunkCoord coord,
    const std::vector<std::uint16_t>& blockIds,
    const std::vector<std::uint8_t>& fluidIds,
    const std::vector<std::uint8_t>& fluidAmounts) const
{
    if (blockIds.size() != kChunkBlockCount ||
        fluidIds.size() != kChunkBlockCount ||
        fluidAmounts.size() != kChunkBlockCount)
    {
        throw std::runtime_error("Cannot build chunk column from invalid voxel data.");
    }

    GeneratedChunkColumn column = generateChunkColumn(coord);
    for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
    {
        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int y = 0; y < kChunkHeight; ++y)
            {
                const std::size_t sourceIndex = chunkBlockIndex(localX, y, localZ);
                const std::uint16_t blockId = blockIds[sourceIndex];
                std::uint8_t fluidId = fluidIds[sourceIndex];
                std::uint8_t fluidAmount = std::min(fluidAmounts[sourceIndex], kMaxFluidAmount);
                if (blockId != kAirBlockId ||
                    fluidId != kWaterFluidId ||
                    fluidAmount == 0)
                {
                    fluidId = kNoFluidId;
                    fluidAmount = 0;
                }
                column.blockAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding) =
                    blockId;
                column.fluidIdAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding) =
                    fluidId;
                column.fluidAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding) =
                    fluidAmount;
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

std::vector<std::uint8_t> WorldGenerator::generateChunkFluids(ChunkCoord coord) const
{
    const GeneratedChunkColumn column = generateChunkColumn(coord);
    std::vector<std::uint8_t> fluidAmounts(kChunkBlockCount);
    for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
    {
        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int y = 0; y < kChunkHeight; ++y)
            {
                fluidAmounts[chunkBlockIndex(localX, y, localZ)] =
                    column.fluidAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding);
            }
        }
    }
    return fluidAmounts;
}

ChunkVoxelData WorldGenerator::generateChunkVoxels(ChunkCoord coord) const
{
    const GeneratedChunkColumn column = generateChunkColumn(coord);
    ChunkVoxelData voxels{};
    voxels.blockIds.resize(kChunkBlockCount);
    voxels.fluidIds.resize(kChunkBlockCount);
    voxels.fluidAmounts.resize(kChunkBlockCount);
    for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
    {
        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int y = 0; y < kChunkHeight; ++y)
            {
                const std::size_t targetIndex = chunkBlockIndex(localX, y, localZ);
                voxels.blockIds[targetIndex] =
                    column.blockAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding);
                voxels.fluidIds[targetIndex] =
                    column.fluidIdAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding);
                voxels.fluidAmounts[targetIndex] =
                    column.fluidAt(localX + GeneratedChunkColumn::kPadding, y, localZ + GeneratedChunkColumn::kPadding);
            }
        }
    }
    return voxels;
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
                else if (y <= kInitialWaterLevel)
                {
                    column.fluidIdAt(localPaddedX, y, localPaddedZ) = kWaterFluidId;
                    column.fluidAt(localPaddedX, y, localPaddedZ) = kMaxFluidAmount;
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

    landformLookup_.enabled = terrainDensityConfig_.landform.enabled;
    landformLookup_.frequency = std::clamp(
        terrainDensityConfig_.landform.frequency,
        0,
        kWorldDensityCellsXZ / 2);
    std::vector<float> rawSamples = std::move(landformLookup_.rawSamples);
    std::vector<float> centerOffsetSamples = std::move(landformLookup_.centerOffsetSamples);
    landformLookup_.rawSamples = std::move(rawSamples);
    landformLookup_.centerOffsetSamples = std::move(centerOffsetSamples);
    landformLookup_.cosX.resize(kWorldDensityCellsXZ);
    landformLookup_.sinX.resize(kWorldDensityCellsXZ);
    landformLookup_.cosZ.resize(kWorldDensityCellsXZ);
    landformLookup_.sinZ.resize(kWorldDensityCellsXZ);

    for (int i = 0; i < kWorldDensityCellsXZ; ++i)
    {
        const float angle = 2.0f * kPi *
            static_cast<float>(landformLookup_.frequency) *
            static_cast<float>(i) /
            static_cast<float>(kWorldDensityCellsXZ);
        landformLookup_.cosX[static_cast<std::size_t>(i)] = std::cos(angle);
        landformLookup_.sinX[static_cast<std::size_t>(i)] = std::sin(angle);
        landformLookup_.cosZ[static_cast<std::size_t>(i)] = landformLookup_.cosX[static_cast<std::size_t>(i)];
        landformLookup_.sinZ[static_cast<std::size_t>(i)] = landformLookup_.sinX[static_cast<std::size_t>(i)];
    }
}

float WorldGenerator::sampleLandformRaw(int cellX, int cellZ) const
{
    if (!landformLookup_.enabled || landformLookup_.frequency == 0)
    {
        return 0.0f;
    }
    if (landformLookup_.cosX.size() != kWorldDensityCellsXZ ||
        landformLookup_.sinX.size() != kWorldDensityCellsXZ ||
        landformLookup_.cosZ.size() != kWorldDensityCellsXZ ||
        landformLookup_.sinZ.size() != kWorldDensityCellsXZ)
    {
        throw std::runtime_error("Landform lookup tables are not initialized.");
    }

    const int wrappedCellX = floorMod(cellX, kWorldDensityCellsXZ);
    const int wrappedCellZ = floorMod(cellZ, kWorldDensityCellsXZ);
    return landformNoise_.sample(
        landformLookup_.cosX.at(static_cast<std::size_t>(wrappedCellX)),
        landformLookup_.sinX.at(static_cast<std::size_t>(wrappedCellX)),
        landformLookup_.cosZ.at(static_cast<std::size_t>(wrappedCellZ)),
        landformLookup_.sinZ.at(static_cast<std::size_t>(wrappedCellZ)));
}

float WorldGenerator::sampleLandformCenterOffset(int cellX, int cellZ) const
{
    if (!landformLookup_.enabled || landformLookup_.frequency == 0 ||
        landformLookup_.rawSamples.size() != landformLookup_.centerOffsetSamples.size() ||
        landformLookup_.centerOffsetSamples.size() < 2)
    {
        return 0.0f;
    }

    const float noiseValue = sampleLandformRaw(cellX, cellZ);
    const auto upperIt = std::lower_bound(
        landformLookup_.rawSamples.begin(),
        landformLookup_.rawSamples.end(),
        noiseValue);
    if (upperIt == landformLookup_.rawSamples.begin())
    {
        return landformLookup_.centerOffsetSamples.front();
    }
    if (upperIt == landformLookup_.rawSamples.end())
    {
        return landformLookup_.centerOffsetSamples.back();
    }

    const std::size_t nextSampleIndex = static_cast<std::size_t>(
        upperIt - landformLookup_.rawSamples.begin());
    const std::size_t sampleIndex = nextSampleIndex - 1;
    const float raw0 = landformLookup_.rawSamples.at(sampleIndex);
    const float raw1 = landformLookup_.rawSamples.at(nextSampleIndex);
    const float t = (noiseValue - raw0) / (raw1 - raw0);
    return lerp(
        landformLookup_.centerOffsetSamples.at(sampleIndex),
        landformLookup_.centerOffsetSamples.at(nextSampleIndex),
        t);
}

float WorldGenerator::sampleDensityLattice(int cellX, int cellY, int cellZ) const
{
    const int wrappedCellX = floorMod(cellX, kWorldDensityCellsXZ);
    const int wrappedCellZ = floorMod(cellZ, kWorldDensityCellsXZ);
    const int clampedCellY = std::clamp(cellY, 0, kWorldDensityCellsY);
    const float worldY = static_cast<float>(clampedCellY * kDensityCellSize);
    const float normalizedY = worldY / static_cast<float>(kChunkHeight);
    const float effectiveCenter = terrainDensityConfig_.gradient.center +
        sampleLandformCenterOffset(cellX, cellZ);
    float density = (effectiveCenter - normalizedY) *
        terrainDensityConfig_.gradient.strength;

    for (const DensityOctave& octave : densityOctaves_)
    {
        if (octave.cosX.size() != kWorldDensityCellsXZ ||
            octave.sinX.size() != kWorldDensityCellsXZ ||
            octave.cosZ.size() != kWorldDensityCellsXZ ||
            octave.sinZ.size() != kWorldDensityCellsXZ)
        {
            throw std::runtime_error("Density octave lookup tables are not initialized.");
        }

        density += noise_.sample(
            octave.cosX.at(static_cast<std::size_t>(wrappedCellX)),
            octave.sinX.at(static_cast<std::size_t>(wrappedCellX)),
            octave.cosZ.at(static_cast<std::size_t>(wrappedCellZ)),
            octave.sinZ.at(static_cast<std::size_t>(wrappedCellZ)),
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

int WorldGenerator::highestSolidYAt(const DensityGrid& densityGrid, int x, int z) const
{
    for (int y = kChunkHeight - 1; y >= 0; --y)
    {
        if (interpolatedDensityAt(densityGrid, x, y, z) >= 0.0f)
        {
            return y;
        }
    }
    return -1;
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
                    blockId = column.fluidIdAt(localPaddedX, y + 1, localPaddedZ) == kWaterFluidId &&
                        column.fluidAt(localPaddedX, y + 1, localPaddedZ) > 0 ?
                        kDirtBlockId :
                        kGrassBlockId;
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

void WorldGenerator::applyPlantDecorations(ChunkCoord coord, GeneratedChunkColumn& column) const
{
    const int minX = coord.x * kChunkSizeX - GeneratedChunkColumn::kPadding;
    const int minZ = coord.z * kChunkSizeZ - GeneratedChunkColumn::kPadding;

    for (int localPaddedZ = 0; localPaddedZ < GeneratedChunkColumn::kDepth; ++localPaddedZ)
    {
        const int z = minZ + localPaddedZ;
        for (int localPaddedX = 0; localPaddedX < GeneratedChunkColumn::kWidth; ++localPaddedX)
        {
            const int x = minX + localPaddedX;
            for (int y = kChunkHeight - 1; y >= 0; --y)
            {
                if (column.blockAt(localPaddedX, y, localPaddedZ) == kAirBlockId)
                {
                    continue;
                }

                if (column.blockAt(localPaddedX, y, localPaddedZ) != kGrassBlockId ||
                    y + 1 >= kChunkHeight ||
                    column.blockAt(localPaddedX, y + 1, localPaddedZ) != kAirBlockId ||
                    column.fluidIdAt(localPaddedX, y + 1, localPaddedZ) != kNoFluidId ||
                    column.fluidAt(localPaddedX, y + 1, localPaddedZ) != 0)
                {
                    break;
                }

                if (treeHash(seed_, x, z, 0x1f8215d5919b6c7dull) % 100 < kPlantSpawnChancePercent)
                {
                    column.blockAt(localPaddedX, y + 1, localPaddedZ) = kPlantBlockId;
                }
                break;
            }
        }
    }
}

void WorldGenerator::applyTreeDecorations(ChunkCoord coord, GeneratedChunkColumn& column) const
{
    const int chunkBaseX = coord.x * kChunkSizeX;
    const int chunkBaseZ = coord.z * kChunkSizeZ;
    const int columnMinX = chunkBaseX - GeneratedChunkColumn::kPadding;
    const int columnMaxX = chunkBaseX + kChunkSizeX - 1 + GeneratedChunkColumn::kPadding;
    const int columnMinZ = chunkBaseZ - GeneratedChunkColumn::kPadding;
    const int columnMaxZ = chunkBaseZ + kChunkSizeZ - 1 + GeneratedChunkColumn::kPadding;
    const int candidateMinX = columnMinX - kTreeDecorationReach;
    const int candidateMaxX = columnMaxX + kTreeDecorationReach;
    const int candidateMinZ = columnMinZ - kTreeDecorationReach;
    const int candidateMaxZ = columnMaxZ + kTreeDecorationReach;
    const DensityGrid densityGrid = buildDensityGrid(candidateMinX, candidateMaxX, candidateMinZ, candidateMaxZ);

    for (int z = candidateMinZ; z <= candidateMaxZ; ++z)
    {
        for (int x = candidateMinX; x <= candidateMaxX; ++x)
        {
            const int highestSolidY = highestSolidYAt(densityGrid, x, z);
            if (highestSolidY < 0 || highestSolidY + 1 <= kInitialWaterLevel)
            {
                continue;
            }

            if (treeHash(seed_, x, z, 0x54b8d7c35f31a21bull) % 100 >= kTreeSpawnChancePercent)
            {
                continue;
            }

            placeTreeInColumn(column, columnMinX, columnMinZ, x, highestSolidY, z);
        }
    }
}

void WorldGenerator::placeTreeInColumn(
    GeneratedChunkColumn& column,
    int minX,
    int minZ,
    int baseX,
    int baseY,
    int baseZ) const
{
    const int trunkHeight = kTreeMinTrunkHeight +
        static_cast<int>(treeHash(seed_, baseX, baseZ, 0x9cd03f1f3f973f01ull) % kTreeTrunkHeightRange);
    const int leafCenterY = baseY + trunkHeight;
    if (leafCenterY + 1 >= kChunkHeight)
    {
        return;
    }

    auto placeLeaf = [&](int x, int y, int z)
    {
        if (y < 0 || y >= kChunkHeight)
        {
            return;
        }

        const int localPaddedX = x - minX;
        const int localPaddedZ = z - minZ;
        if (localPaddedX < 0 || localPaddedX >= GeneratedChunkColumn::kWidth ||
            localPaddedZ < 0 || localPaddedZ >= GeneratedChunkColumn::kDepth)
        {
            return;
        }
        const std::uint16_t blockId = column.blockAt(localPaddedX, y, localPaddedZ);
        if ((blockId != kAirBlockId && blockId != kPlantBlockId) ||
            column.fluidIdAt(localPaddedX, y, localPaddedZ) != kNoFluidId ||
            column.fluidAt(localPaddedX, y, localPaddedZ) != 0)
        {
            return;
        }

        column.blockAt(localPaddedX, y, localPaddedZ) = kLeavesBlockId;
    };

    auto placeTrunk = [&](int x, int y, int z)
    {
        if (y < 0 || y >= kChunkHeight)
        {
            return;
        }

        const int localPaddedX = x - minX;
        const int localPaddedZ = z - minZ;
        if (localPaddedX < 0 || localPaddedX >= GeneratedChunkColumn::kWidth ||
            localPaddedZ < 0 || localPaddedZ >= GeneratedChunkColumn::kDepth)
        {
            return;
        }

        const std::uint16_t blockId = column.blockAt(localPaddedX, y, localPaddedZ);
        if (blockId != kAirBlockId &&
            blockId != kPlantBlockId &&
            blockId != kLeavesBlockId &&
            blockId != kTrunkBlockId)
        {
            return;
        }
        if (column.fluidIdAt(localPaddedX, y, localPaddedZ) != kNoFluidId ||
            column.fluidAt(localPaddedX, y, localPaddedZ) != 0)
        {
            return;
        }

        column.blockAt(localPaddedX, y, localPaddedZ) = kTrunkBlockId;
    };

    for (int layer = -2; layer <= 1; ++layer)
    {
        const int radius = layer == 1 ? 1 : 2;
        const int y = leafCenterY + layer;
        for (int dz = -radius; dz <= radius; ++dz)
        {
            for (int dx = -radius; dx <= radius; ++dx)
            {
                if (std::abs(dx) + std::abs(dz) > radius + 1)
                {
                    continue;
                }
                if (radius == 2 &&
                    std::abs(dx) == radius &&
                    std::abs(dz) == radius &&
                    treeHash(seed_, baseX + dx, baseZ + dz, static_cast<std::uint64_t>(y)) % 100 < 45)
                {
                    continue;
                }

                placeLeaf(baseX + dx, y, baseZ + dz);
            }
        }
    }

    for (int offsetY = 1; offsetY <= trunkHeight; ++offsetY)
    {
        placeTrunk(baseX, baseY + offsetY, baseZ);
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
        if (blockOverride.blockId != kAirBlockId)
        {
            column.fluidIdAt(localPaddedX, blockOverride.y, localPaddedZ) = kNoFluidId;
            column.fluidAt(localPaddedX, blockOverride.y, localPaddedZ) = 0;
        }
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
