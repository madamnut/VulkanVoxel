#pragma once

#include "MathTypes.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
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

struct BlockRaycastHit {
    bool hit = false;
    int blockX = 0;
    int blockY = 0;
    int blockZ = 0;
    int placeX = 0;
    int placeY = 0;
    int placeZ = 0;
};

class VoxelWorld {
public:
    static bool IsInsideWorld(int worldX, int worldY, int worldZ);
    void EnsureChunkColumn(int chunkX, int chunkZ);
    void EnsureRange(int minChunkX, int maxChunkX, int minChunkZ, int maxChunkZ);
    std::uint8_t GetBlock(int worldX, int worldY, int worldZ);
    bool SetBlock(int worldX, int worldY, int worldZ, std::uint8_t blockValue);
    bool IsChunkColumnModified(int chunkX, int chunkZ) const;
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
    static std::size_t GetBlockIndex(int worldX, int worldY, int worldZ);
    static int GetSubChunkIndex(int worldY);
    static int GetSubChunkBlockIndex(int localX, int localY, int localZ);

    void EnsureInitialized();
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
    void AppendFaceQuad(
        WorldMeshData& mesh,
        int worldX,
        int worldY,
        int worldZ,
        int faceIndex
    ) const;
    void AppendDetailedColumnMesh(WorldMeshData& mesh, int chunkX, int chunkZ);
    void MarkChunkColumnModified(int chunkX, int chunkZ);

    std::vector<std::uint8_t> blocks_;
    bool initialized_ = false;
    std::unordered_set<std::int64_t> modifiedColumns_;
};
