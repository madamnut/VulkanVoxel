#pragma once

#include "world/Block.h"
#include "world/ChunkTypes.h"
#include "world/SimplexNoise.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

constexpr int kTerrainDirtDepth = 4;
constexpr int kMaxTerrainDensityOctaves = 8;

struct DensityGradientConfig
{
    float center = 0.5f;
    float strength = 512.0f;
};

struct DensityNoiseConfig
{
    int octaves = 3;
    int baseFrequency = 4;
    float frequencyMultiplier = 4.0f;
    float baseAmplitude = 48.0f;
    float amplitudeMultiplier = 0.45f;
    float verticalFrequencyScale = 1.0f;
};

struct TerrainDensityConfig
{
    DensityGradientConfig gradient{};
    DensityNoiseConfig noise{};
};

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
    WorldGenerator();

    void setSeed(std::uint64_t seed);
    std::uint64_t seed() const;
    void setTerrainDensityConfig(const TerrainDensityConfig& config);

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
    struct DensityOctave
    {
        int frequencyXZ = 1;
        float frequencyY = 0.0f;
        float amplitude = 0.0f;
        std::vector<float> cosX;
        std::vector<float> sinX;
        std::vector<float> cosZ;
        std::vector<float> sinZ;
        std::array<float, kWorldDensityVerticesY> yInput{};
    };

    struct DensityGrid
    {
        int minCellX = 0;
        int minCellZ = 0;
        int countX = 0;
        int countZ = 0;
        std::vector<float> values;

        float valueAt(int localCellX, int cellY, int localCellZ) const;
    };

    struct BlockOverride
    {
        int x = 0;
        int y = 0;
        int z = 0;
        std::uint16_t blockId = kAirBlockId;
    };

    static std::int64_t blockKey(int x, int y, int z);
    static int floorDiv(int value, int divisor);
    static int floorMod(int value, int divisor);
    static float lerp(float a, float b, float t);
    void rebuildNoiseLookups();
    float sampleDensityLattice(int cellX, int cellY, int cellZ) const;
    DensityGrid buildDensityGrid(int minBlockX, int maxBlockX, int minBlockZ, int maxBlockZ) const;
    float interpolatedDensityAt(const DensityGrid& densityGrid, int x, int y, int z) const;
    void generateBaseTerrain(ChunkCoord coord, GeneratedChunkColumn& column) const;
    void applySurfaceMaterials(GeneratedChunkColumn& column) const;
    void applyBlockOverrides(ChunkCoord coord, GeneratedChunkColumn& column) const;

    std::uint64_t seed_ = 0;
    SimplexNoise5D noise_{};
    TerrainDensityConfig terrainDensityConfig_{};
    std::vector<DensityOctave> densityOctaves_;
    mutable std::shared_mutex overrideMutex_;
    std::unordered_map<std::int64_t, BlockOverride> blockOverrides_;
};
