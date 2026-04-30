#pragma once

#include "world/ChunkMesher.h"
#include "world/ChunkTypes.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ChunkLoadUpdateStats
{
    std::size_t loaded = 0;
    std::size_t unloaded = 0;
    std::size_t queued = 0;
};

class ChunkStreamingManager
{
public:
    explicit ChunkStreamingManager(ChunkMesher& chunkMesher);
    ~ChunkStreamingManager();

    void setLoadRadius(int loadRadius);
    void setBuildThreadCount(int buildThreadCount);
    void setChunkBuildCallback(std::function<std::shared_ptr<ChunkBuildResult>(ChunkCoord, std::uint64_t)> callback);
    void setBuildErrorCallback(std::function<void(ChunkCoord, const std::string&)> callback);
    int loadRadius() const;
    int buildThreadCount() const;

    void startWorkers();
    void stopWorkers();
    void reset();

    ChunkLoadUpdateStats updateLoadedChunks(
        int centerChunkX,
        int centerChunkZ,
        const std::function<void(ChunkCoord)>& unloadChunk);

    std::size_t processCompletedSubchunkBuilds();
    std::shared_ptr<ChunkBuildResult> popCompletedChunkBuild();
    bool shouldAcceptCompletedChunk(ChunkCoord coord) const;
    void markChunkLoaded(ChunkCoord coord);
    bool isChunkLoaded(ChunkCoord coord) const;
    bool rebuildLoadedChunk(ChunkCoord coord);

    std::size_t loadedChunkCount() const;
    std::pair<std::size_t, std::size_t> queueSizes() const;

private:
    struct PendingChunkMesh
    {
        std::uint64_t generation = 0;
        int completedSubchunks = 0;
        std::array<bool, kSubchunksPerChunk> received{};
        std::array<std::vector<BlockVertex>, kSubchunksPerChunk> vertices;
        std::array<std::vector<std::uint32_t>, kSubchunksPerChunk> indices;
        std::array<std::vector<BlockVertex>, kSubchunksPerChunk> fluidVertices;
        std::array<std::vector<std::uint32_t>, kSubchunksPerChunk> fluidIndices;
    };

    ChunkMesher& chunkMesher_;
    std::function<std::shared_ptr<ChunkBuildResult>(ChunkCoord, std::uint64_t)> chunkBuildCallback_;
    std::function<void(ChunkCoord, const std::string&)> buildErrorCallback_;
    int loadRadius_ = 5;
    int buildThreadCount_ = 4;

    void workerLoop();
    void queueChunkBuildLocked(ChunkCoord coord, int centerChunkX, int centerChunkZ);
    void reprioritizePendingChunkBuildsLocked(int centerChunkX, int centerChunkZ);
    std::size_t cancelQueuedChunksOutsideDesiredLocked();
    bool isChunkInLoadRange(ChunkCoord coord, int centerChunkX, int centerChunkZ) const;
    bool isChunkInCurrentLoadRangeLocked(ChunkCoord coord) const;
    std::shared_ptr<ChunkBuildResult> assembleChunkMesh(ChunkCoord coord, PendingChunkMesh& pendingMesh) const;

    std::unordered_set<ChunkCoord, ChunkCoordHash> desiredChunks_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> loadedChunks_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> queuedChunks_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> rebuildingChunks_;
    std::vector<std::thread> workers_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<ChunkBuildRequest> pendingBuilds_;
    std::deque<SubchunkBuildResult> completedSubchunks_;
    std::deque<std::shared_ptr<ChunkBuildResult>> completedChunks_;
    bool workerRunning_ = false;
    std::uint64_t buildGeneration_ = 0;
    int priorityCenterChunkX_ = 0;
    int priorityCenterChunkZ_ = 0;
    int loadedCenterChunkX_ = std::numeric_limits<int>::min();
    int loadedCenterChunkZ_ = std::numeric_limits<int>::min();
    std::unordered_map<ChunkCoord, std::uint64_t, ChunkCoordHash> queuedGenerations_;
    std::unordered_map<ChunkCoord, PendingChunkMesh, ChunkCoordHash> pendingChunkMeshes_;
};
