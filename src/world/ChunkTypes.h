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
