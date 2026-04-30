#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr int kChunkSizeX = 16;
constexpr int kChunkSizeZ = 16;
constexpr int kSubchunkSize = 16;
constexpr int kChunkHeight = 512;
constexpr int kSubchunksPerChunk = kChunkHeight / kSubchunkSize;
constexpr int kWorldSizeXZ = 65536;
constexpr int kWorldChunkSide = kWorldSizeXZ / kChunkSizeX;
constexpr int kDensityCellSize = 4;
constexpr int kWorldDensityCellsXZ = kWorldSizeXZ / kDensityCellSize;
constexpr int kWorldDensityCellsY = kChunkHeight / kDensityCellSize;
constexpr int kWorldDensityVerticesY = kWorldDensityCellsY + 1;
constexpr std::uint16_t kMaxFluidAmount = 100;
constexpr std::uint16_t kNoFluidState = 0;
constexpr std::uint16_t kWaterFluidType = 1;
constexpr std::uint16_t kWaterFullFluidState = kMaxFluidAmount;
constexpr int kInitialWaterLevel = 205;
constexpr std::size_t kChunkBlockCount =
    static_cast<std::size_t>(kChunkSizeX) * kChunkHeight * kChunkSizeZ;

inline std::uint16_t encodeFluidState(std::uint16_t fluidType, std::uint16_t amount)
{
    if (fluidType == 0 || amount == 0)
    {
        return kNoFluidState;
    }
    const std::uint16_t clampedAmount = amount > kMaxFluidAmount ? kMaxFluidAmount : amount;
    return static_cast<std::uint16_t>((fluidType - 1) * kMaxFluidAmount + clampedAmount);
}

inline std::uint16_t fluidTypeFromState(std::uint16_t fluidState)
{
    return fluidState == kNoFluidState
        ? 0
        : static_cast<std::uint16_t>((fluidState - 1) / kMaxFluidAmount + 1);
}

inline std::uint16_t fluidAmountFromState(std::uint16_t fluidState)
{
    return fluidState == kNoFluidState
        ? 0
        : static_cast<std::uint16_t>((fluidState - 1) % kMaxFluidAmount + 1);
}

inline bool isWaterFluidState(std::uint16_t fluidState)
{
    return fluidTypeFromState(fluidState) == kWaterFluidType;
}

inline std::size_t chunkBlockIndex(int localX, int y, int localZ)
{
    return static_cast<std::size_t>((localZ * kChunkSizeX + localX) * kChunkHeight + y);
}

struct BlockVertex
{
    float position[3];
    float uv[2];
    float ao = 1.0f;
    float textureLayer = 0.0f;
};

struct MeshRange
{
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::int32_t vertexOffset = 0;
};

struct SubchunkDraw
{
    int chunkX = 0;
    int chunkZ = 0;
    int subchunkY = 0;
    MeshRange range{};
};

struct ChunkCoord
{
    int x = 0;
    int z = 0;

    bool operator==(ChunkCoord other) const
    {
        return x == other.x && z == other.z;
    }
};

struct ChunkCoordHash
{
    std::size_t operator()(ChunkCoord coord) const
    {
        const std::uint64_t x = static_cast<std::uint32_t>(coord.x);
        const std::uint64_t z = static_cast<std::uint32_t>(coord.z);
        return static_cast<std::size_t>((x << 32) ^ z);
    }
};

struct ChunkBuildRequest
{
    ChunkCoord coord{};
    int subchunkY = 0;
    std::int64_t priorityDistanceSq = 0;
    std::uint64_t generation = 0;
};

struct SubchunkBuildResult
{
    ChunkCoord coord{};
    int subchunkY = 0;
    std::uint64_t generation = 0;
    std::vector<BlockVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<BlockVertex> fluidVertices;
    std::vector<std::uint32_t> fluidIndices;
};

struct ChunkBuildProfile
{
    double totalMs = 0.0;
    double dataMs = 0.0;
    double dataLoadedLookupMs = 0.0;
    double dataCacheLookupMs = 0.0;
    double dataWaitMs = 0.0;
    double dataLoadGenerateMs = 0.0;
    double dataCopyMs = 0.0;
    double dataCacheStoreMs = 0.0;
    double diskLoadMs = 0.0;
    double generateMs = 0.0;
    double genLockMs = 0.0;
    double genDensityGridMs = 0.0;
    double genBaseTerrainMs = 0.0;
    double genSurfaceMs = 0.0;
    double genPlantMs = 0.0;
    double genTreeMs = 0.0;
    double genOverrideMs = 0.0;
    double genVoxelCopyMs = 0.0;
    double diskSaveMs = 0.0;
    double columnMs = 0.0;
    double meshMs = 0.0;
    std::uint32_t loadedHits = 0;
    std::uint32_t cacheHits = 0;
    std::uint32_t waitedLoads = 0;
    std::uint32_t diskLoaded = 0;
    std::uint32_t generated = 0;
};

struct ChunkBuildResult
{
    ChunkBuildResult() = default;

    ChunkCoord coord{};
    ChunkBuildProfile profile{};
    std::vector<BlockVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<SubchunkDraw> subchunks;
    std::vector<BlockVertex> fluidVertices;
    std::vector<std::uint32_t> fluidIndices;
    std::vector<SubchunkDraw> fluidSubchunks;
};

struct LoadedChunkData
{
    ChunkCoord coord{};
    std::vector<std::uint16_t> blockIds;
    std::vector<std::uint16_t> fluidStates;
    bool dirty = false;
};

struct ChunkVoxelData
{
    std::vector<std::uint16_t> blockIds;
    std::vector<std::uint16_t> fluidStates;

    bool valid() const
    {
        return blockIds.size() == kChunkBlockCount &&
            fluidStates.size() == kChunkBlockCount;
    }
};
