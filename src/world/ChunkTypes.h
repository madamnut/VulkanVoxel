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
constexpr std::uint8_t kMaxFluidAmount = 100;
constexpr std::uint8_t kNoFluidId = 0;
constexpr std::uint8_t kWaterFluidId = 1;
constexpr int kInitialWaterLevel = 205;
constexpr std::size_t kChunkBlockCount =
    static_cast<std::size_t>(kChunkSizeX) * kChunkHeight * kChunkSizeZ;

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
};

struct ChunkBuildResult
{
    ChunkCoord coord{};
    std::vector<BlockVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<SubchunkDraw> subchunks;
    std::vector<std::uint16_t> blockIds;
    std::vector<std::uint8_t> fluidIds;
    std::vector<std::uint8_t> fluidAmounts;
};

struct LoadedChunkData
{
    ChunkCoord coord{};
    std::vector<std::uint16_t> blockIds;
    std::vector<std::uint8_t> fluidIds;
    std::vector<std::uint8_t> fluidAmounts;
    bool dirty = false;
};

struct ChunkVoxelData
{
    std::vector<std::uint16_t> blockIds;
    std::vector<std::uint8_t> fluidIds;
    std::vector<std::uint8_t> fluidAmounts;

    bool valid() const
    {
        return blockIds.size() == kChunkBlockCount &&
            fluidIds.size() == kChunkBlockCount &&
            fluidAmounts.size() == kChunkBlockCount;
    }
};
