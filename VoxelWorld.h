#pragma once

#include "MathTypes.h"
#include "TerrainConfig.h"

#include <cstdint>
#include <filesystem>
#include <deque>
#include <istream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

constexpr int kWorldSizeY = 512;

constexpr int kChunkSizeX = 16;
constexpr int kChunkSizeZ = 16;
constexpr int kSubChunkSize = 16;
constexpr int kSubChunkCountY = kWorldSizeY / kSubChunkSize;
constexpr int kSubChunkVoxelCount = kSubChunkSize * kSubChunkSize * kSubChunkSize;
constexpr int kRegionSizeInChunks = 32;

struct WorldVertex {
    float position[3];
    float uv[2];
    float ao = 1.0f;
};

struct WorldQuadRecord {
    std::uint32_t packed0 = 0;
    std::uint32_t packed1 = 0;
};

struct WorldBatchId {
    int chunkX = 0;
    int chunkZ = 0;
    int subChunkIndex = 0;

    bool operator==(const WorldBatchId& other) const = default;
};

struct WorldBatchIdHash {
    std::size_t operator()(const WorldBatchId& id) const noexcept {
        const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.chunkX));
        const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.chunkZ));
        const std::uint64_t s = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.subChunkIndex));
        return static_cast<std::size_t>((x * 73856093ull) ^ (z * 19349663ull) ^ (s * 83492791ull));
    }
};

struct WorldVisibleBatch {
    WorldBatchId id{};
    std::uint64_t revision = 0;
};

struct SubChunkMeshData {
    std::vector<WorldQuadRecord> quads;
    bool dirty = true;
    std::uint64_t revision = 0;
};

struct SubChunkVoxelData {
    std::vector<std::uint16_t> blocks;
    std::uint16_t uniformBlock = 0;
    bool isUniform = true;
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

struct ChunkColumnData {
    std::vector<SubChunkVoxelData> subChunks;
    std::vector<SubChunkMeshData> subChunkMeshes;
    bool modified = false;
};

struct DirtySubChunkId {
    int chunkX = 0;
    int chunkZ = 0;
    int subChunkIndex = 0;

    bool operator==(const DirtySubChunkId& other) const = default;
};

struct DirtySubChunkIdHash {
    std::size_t operator()(const DirtySubChunkId& id) const noexcept {
        const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.chunkX));
        const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.chunkZ));
        const std::uint64_t s = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.subChunkIndex));
        return static_cast<std::size_t>((x * 73856093ull) ^ (z * 19349663ull) ^ (s * 83492791ull));
    }
};

struct PendingChunkId {
    int chunkX = 0;
    int chunkZ = 0;

    bool operator==(const PendingChunkId& other) const = default;
};

struct PendingChunkIdHash {
    std::size_t operator()(const PendingChunkId& id) const noexcept {
        const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.chunkX));
        const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id.chunkZ));
        return static_cast<std::size_t>((x * 73856093ull) ^ (z * 19349663ull));
    }
};

struct ChunkMeshBatchData {
    PendingChunkId id{};
    std::vector<WorldQuadRecord> quads;
};

struct WorldMeshData {
    std::vector<WorldVisibleBatch> batches;
    std::size_t loadedChunkCount = 0;
    std::uint32_t totalVertexCount = 0;
    std::uint32_t totalIndexCount = 0;
};

struct WorldRenderUpdate {
    std::vector<ChunkMeshBatchData> uploads;
    std::vector<PendingChunkId> removals;
    std::size_t loadedChunkCount = 0;

    bool HasChanges() const {
        return !uploads.empty() || !removals.empty();
    }
};

struct PreparedChunkColumn {
    PendingChunkId id{};
    ChunkColumnData column;
    bool generated = false;
};

struct MeshBuildInput {
    DirtySubChunkId id{};
    int minWorldX = 0;
    int minWorldY = 0;
    int minWorldZ = 0;
    std::vector<std::uint16_t> blocks;
};

struct PreparedSubChunkMesh {
    DirtySubChunkId id{};
    std::vector<WorldQuadRecord> quads;
};

struct RegionSaveChunkSnapshot {
    PendingChunkId id{};
    std::uint64_t generation = 0;
    ChunkColumnData column;
};

struct RegionSaveTask {
    int regionX = 0;
    int regionZ = 0;
    std::vector<RegionSaveChunkSnapshot> chunks;
};

struct RegionChunkIndexMetadata {
    std::int32_t chunkX = 0;
    std::int32_t chunkZ = 0;
    std::uint64_t offset = 0;
    std::uint32_t size = 0;
};

struct RegionIndexCacheEntry {
    bool regionExists = false;
    std::unordered_map<std::int64_t, RegionChunkIndexMetadata> chunkIndex;
};

struct RuntimeProfileStage {
    double averageMs = 0.0;
    double maxMs = 0.0;
    std::uint32_t samples = 0;
};

struct VoxelWorldRuntimeProfileSnapshot {
    RuntimeProfileStage chunkLoad{};
    RuntimeProfileStage diskLoad{};
    RuntimeProfileStage generate{};
    RuntimeProfileStage meshBuild{};
    RuntimeProfileStage save{};
};

VoxelWorldRuntimeProfileSnapshot ConsumeVoxelWorldRuntimeProfileSnapshot();

class VoxelWorld {
public:
    VoxelWorld();
    ~VoxelWorld();
    VoxelWorld(VoxelWorld&&) noexcept;
    VoxelWorld& operator=(VoxelWorld&&) noexcept;
    VoxelWorld(const VoxelWorld&) = delete;
    VoxelWorld& operator=(const VoxelWorld&) = delete;

    static bool IsInsideWorld(int worldX, int worldY, int worldZ);
    void UpdateStreamingTargets(int centerChunkX, int centerChunkZ, int keepRadius);
    std::vector<PendingChunkId> AcquireChunkLoadRequests(std::size_t maxCount);
    PreparedChunkColumn PrepareChunkColumn(int chunkX, int chunkZ) const;
    bool CommitPreparedChunkColumn(PreparedChunkColumn&& prepared);
    std::vector<MeshBuildInput> AcquireDirtyMeshRequests(std::size_t maxCount, int centerChunkX, int centerChunkZ, int keepRadius);
    PreparedSubChunkMesh PrepareSubChunkMesh(const MeshBuildInput& input) const;
    bool CommitPreparedSubChunkMesh(PreparedSubChunkMesh&& preparedMesh);
    WorldRenderUpdate DrainRenderUpdates();
    bool HasPendingRenderUpdates() const;
    std::size_t FinalizeStreamingWindow(int centerChunkX, int centerChunkZ, int keepRadius, std::size_t unloadBudget = static_cast<std::size_t>(-1));
    std::size_t UpdateStreamingWindow(int centerChunkX, int centerChunkZ, int keepRadius, std::size_t generationBudget = static_cast<std::size_t>(-1));
    void EnsureChunkColumn(int chunkX, int chunkZ);
    void EnsureRange(int minChunkX, int maxChunkX, int minChunkZ, int maxChunkZ);
    std::uint16_t GetBlock(int worldX, int worldY, int worldZ);
    bool SetBlock(int worldX, int worldY, int worldZ, std::uint16_t blockValue);
    void Save();
    bool CopyChunkMeshBatch(int chunkX, int chunkZ, ChunkMeshBatchData& outBatch) const;
    std::vector<RegionSaveTask> DrainPendingSaveTasks(std::size_t maxCount);
    void CompletePendingSaveTask(const RegionSaveTask& completedTask);
    void ExecuteSaveTask(const RegionSaveTask& task) const;
    bool HasPendingSaveTasks() const;
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
    const TerrainConfig& GetTerrainConfig() const;

private:
    static std::int64_t MakeChunkKey(int chunkX, int chunkZ);
    static int FloorDiv(int value, int divisor);
    static int PositiveMod(int value, int divisor);
    static int GetRegionCoord(int chunkCoord);
    static int GetRegionLocalCoord(int chunkCoord);
    static std::int64_t MakeRegionKey(int regionX, int regionZ);
    static int GetSubChunkIndex(int worldY);
    static int GetSubChunkBlockIndex(int localX, int localY, int localZ);
    static std::uint16_t GetSubChunkBlock(const SubChunkVoxelData& subChunk, int localX, int localY, int localZ);
    static void SetSubChunkBlock(SubChunkVoxelData& subChunk, int localX, int localY, int localZ, std::uint16_t blockValue);
    static void TryCollapseSubChunk(SubChunkVoxelData& subChunk);
    std::uint16_t SampleGeneratedBlock(int worldX, int worldY, int worldZ) const;

    void EnsureInitialized();
    void InitializeVoxelSubChunks(ChunkColumnData& column) const;
    void InitializeSubChunkMeshes(ChunkColumnData& column, bool dirty) const;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> CollectDesiredChunks(int centerChunkX, int centerChunkZ, int keepRadius) const;
    void RebuildPendingChunkQueue(int centerChunkX, int centerChunkZ);
    void RefreshStreamingQueue(int centerChunkX, int centerChunkZ, int keepRadius);
    void EnqueueDirtySubChunk(int chunkX, int chunkZ, int subChunkIndex);
    void EnqueueAllDirtySubChunks(int chunkX, int chunkZ, ChunkColumnData& column);
    void RemoveQueuedDirtySubChunksForChunk(int chunkX, int chunkZ);
    void MarkSubChunkDirty(int chunkX, int chunkZ, int subChunkIndex);
    bool IsChunkDesired(int chunkX, int chunkZ) const;
    void QueueRenderChunkUpdate(int chunkX, int chunkZ);
    void QueueRenderChunkRemoval(int chunkX, int chunkZ);
    void QueueRenderUpdatesForChunk(int chunkX, int chunkZ, const ChunkColumnData& column);
    std::size_t RebuildDirtyMeshesInWindow(int centerChunkX, int centerChunkZ, int keepRadius);
    void UnloadRetiredChunks(std::size_t unloadBudget = static_cast<std::size_t>(-1));
    void GenerateChunkColumn(int chunkX, int chunkZ, ChunkColumnData& outColumn) const;
    void LoadOrCreateSave();
    void SaveAllDirtyChunks();
    bool LoadChunkColumn(int chunkX, int chunkZ, ChunkColumnData& outColumn) const;
    void SaveChunkColumn(int chunkX, int chunkZ, ChunkColumnData& column);
    std::vector<std::uint8_t> SerializeChunkColumnPayload(const ChunkColumnData& column) const;
    void DeserializeChunkColumnPayload(std::istream& stream, ChunkColumnData& column) const;
    void SaveRegionOverrides(int regionX, int regionZ, const std::unordered_map<std::int64_t, const ChunkColumnData*>& overrides) const;
    void LoadRegionFile(int regionX, int regionZ, std::unordered_map<std::int64_t, ChunkColumnData>& outColumns) const;
    void SaveRegionFile(int regionX, int regionZ, const std::unordered_map<std::int64_t, ChunkColumnData>& columns) const;
    bool TryGetRegionChunkIndexMetadata(int regionX, int regionZ, int chunkX, int chunkZ, RegionChunkIndexMetadata& outMetadata) const;
    void RefreshRegionIndexCacheEntry(int regionX, int regionZ, const std::filesystem::path& regionPath) const;
    void InvalidateRegionIndexCacheEntry(int regionX, int regionZ) const;
    bool HasPendingMeshWorkForChunk(int chunkX, int chunkZ) const;
    static bool IsChunkInsideFrustum(
        int chunkX,
        int chunkZ,
        const Vec3& cameraPosition,
        const Vec3& cameraForward,
        float verticalFovDegrees,
        float aspectRatio
    );
    void RebuildSubChunkMesh(int chunkX, int chunkZ, int subChunkIndex);
    void EnsureTerrainConfigLoaded();
    static std::string GetSaveRootPath();
    static std::string GetLevelFilePath();
    static std::string GetRegionDirectoryPath();
    static std::string GetRegionFilePath(int regionX, int regionZ);

    bool initialized_ = false;
    bool terrainConfigLoaded_ = false;
    bool saveDirty_ = false;
    bool streamingWindowInitialized_ = false;
    int streamingCenterChunkX_ = 0;
    int streamingCenterChunkZ_ = 0;
    int streamingKeepRadius_ = 0;
    TerrainConfig terrainConfig_{}; 
    std::unordered_map<std::int64_t, ChunkColumnData> chunkColumns_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> desiredChunks_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> retiringChunks_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> inFlightChunkLoads_;
    std::deque<PendingChunkId> pendingChunkLoadQueue_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> queuedChunkLoads_;
    std::deque<DirtySubChunkId> dirtySubChunkQueue_;
    std::unordered_set<DirtySubChunkId, DirtySubChunkIdHash> queuedDirtySubChunks_;
    std::unordered_set<DirtySubChunkId, DirtySubChunkIdHash> inFlightDirtySubChunks_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> activeRenderChunks_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> pendingRenderChunkUpdates_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> pendingRenderChunkRemovals_;
    bool renderStatsDirty_ = true;
    struct PendingSavedChunkState {
        std::uint64_t generation = 0;
        ChunkColumnData column;
    };
    mutable std::recursive_mutex regionFileIoMutex_;
    mutable std::mutex pendingSaveMutex_;
    std::deque<RegionSaveTask> pendingSaveTasks_;
    std::unordered_map<std::int64_t, PendingSavedChunkState> pendingSavedChunkColumns_;
    std::uint64_t nextPendingSaveGeneration_ = 1;
    mutable std::mutex regionIndexCacheMutex_;
    mutable std::unordered_map<std::int64_t, RegionIndexCacheEntry> regionIndexCache_;
};
