#pragma once

#include "MathTypes.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

constexpr int kWorldSizeX = 512;
constexpr int kWorldSizeY = 512;
constexpr int kWorldSizeZ = 512;
constexpr int kGroundHeight = 256;

constexpr int kChunkSizeX = 16;
constexpr int kChunkSizeZ = 16;
constexpr int kSubChunkSize = 16;
constexpr int kSubChunkCountY = kWorldSizeY / kSubChunkSize;
constexpr int kWorldChunkCountX = kWorldSizeX / kChunkSizeX;
constexpr int kWorldChunkCountZ = kWorldSizeZ / kChunkSizeZ;

struct WorldVertex {
    float position[3];
    float uv[2];
};

struct OverlayVertex {
    float position[2];
    float color[3];
};

struct SubChunk {
    std::array<std::uint8_t, kSubChunkSize * kSubChunkSize * kSubChunkSize> blocks{};
    bool isEmpty = true;
    bool isFull = false;
};

struct ChunkColumn {
    int chunkX = 0;
    int chunkZ = 0;
    std::array<SubChunk, kSubChunkCountY> subChunks{};
};

class VoxelWorld {
public:
    void EnsureChunkColumn(int chunkX, int chunkZ);
    void EnsureRange(int minChunkX, int maxChunkX, int minChunkZ, int maxChunkZ);
    void BuildVisibleMesh(int centerChunkX, int centerChunkZ, int renderRadius, std::vector<WorldVertex>& outVertices);
    std::size_t GetLoadedChunkCount() const;

private:
    static std::int64_t MakeChunkKey(int chunkX, int chunkZ);
    static int GetSubChunkIndex(int worldY);
    static int GetSubChunkBlockIndex(int localX, int localY, int localZ);

    void GenerateChunkColumn(ChunkColumn& column) const;
    void AppendTopFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const;
    void AppendNorthFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const;
    void AppendSouthFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const;
    void AppendWestFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const;
    void AppendEastFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const;

    std::unordered_map<std::int64_t, ChunkColumn> columns_;
};
