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

struct WorldMeshData {
    std::vector<WorldVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::size_t loadedChunkCount = 0;
};

struct OverlayVertex {
    float position[2];
    float uv[2];
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

struct FontGlyphBitmap {
    int width = 0;
    int height = 0;
    int offsetX = 0;
    int offsetY = 0;
    int advance = 0;
    bool valid = false;
    float u0 = 0.0f;
    float v0 = 0.0f;
    float u1 = 0.0f;
    float v1 = 0.0f;
    std::vector<std::uint8_t> alpha;
};

class VoxelWorld {
public:
    void EnsureChunkColumn(int chunkX, int chunkZ);
    void EnsureRange(int minChunkX, int maxChunkX, int minChunkZ, int maxChunkZ);
    void BuildVisibleMesh(
        int centerChunkX,
        int centerChunkZ,
        int renderRadius,
        const Vec3& cameraPosition,
        const Vec3& cameraForward,
        float verticalFovDegrees,
        float aspectRatio,
        WorldMeshData& outMesh
    );
    std::size_t GetLoadedChunkCount() const;

private:
    static std::int64_t MakeChunkKey(int chunkX, int chunkZ);
    static int GetSubChunkIndex(int worldY);
    static int GetSubChunkBlockIndex(int localX, int localY, int localZ);

    void GenerateChunkColumn(ChunkColumn& column) const;
    static bool IsChunkInsideFrustum(
        int chunkX,
        int chunkZ,
        const Vec3& cameraPosition,
        const Vec3& cameraForward,
        float verticalFovDegrees,
        float aspectRatio
    );
    void AppendTopQuad(WorldMeshData& mesh, int startX, int startZ, int width, int depth) const;
    void AppendNorthQuad(WorldMeshData& mesh, int startX, int width) const;
    void AppendSouthQuad(WorldMeshData& mesh, int startX, int width) const;
    void AppendWestQuad(WorldMeshData& mesh, int startZ, int depth) const;
    void AppendEastQuad(WorldMeshData& mesh, int startZ, int depth) const;

    std::unordered_map<std::int64_t, ChunkColumn> columns_;
};
