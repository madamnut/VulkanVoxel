#include "world/ChunkStreamingManager.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
std::int64_t chunkDistanceSq(ChunkCoord coord, int centerChunkX, int centerChunkZ)
{
    const std::int64_t dx = static_cast<std::int64_t>(coord.x) - centerChunkX;
    const std::int64_t dz = static_cast<std::int64_t>(coord.z) - centerChunkZ;
    return dx * dx + dz * dz;
}

struct ChunkBuildRequestPriority
{
    bool operator()(const ChunkBuildRequest& lhs, const ChunkBuildRequest& rhs) const
    {
        if (lhs.priorityDistanceSq != rhs.priorityDistanceSq)
        {
            return lhs.priorityDistanceSq > rhs.priorityDistanceSq;
        }
        return lhs.generation < rhs.generation;
    }
};
}

ChunkStreamingManager::ChunkStreamingManager(ChunkMesher& chunkMesher)
    : chunkMesher_(chunkMesher)
{
}

ChunkStreamingManager::~ChunkStreamingManager()
{
    stopWorkers();
}

void ChunkStreamingManager::setLoadRadius(int loadRadius)
{
    loadRadius_ = loadRadius;
}

void ChunkStreamingManager::setBuildThreadCount(int buildThreadCount)
{
    buildThreadCount_ = buildThreadCount;
}

int ChunkStreamingManager::loadRadius() const
{
    return loadRadius_;
}

int ChunkStreamingManager::buildThreadCount() const
{
    return buildThreadCount_;
}

void ChunkStreamingManager::startWorkers()
{
    workerRunning_ = true;
    workers_.reserve(static_cast<std::size_t>(buildThreadCount_));
    for (int i = 0; i < buildThreadCount_; ++i)
    {
        workers_.emplace_back([this]()
        {
            workerLoop();
        });
    }
}

void ChunkStreamingManager::stopWorkers()
{
    if (!workerRunning_ && workers_.empty())
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        workerRunning_ = false;
        pendingBuilds_.clear();
        completedSubchunks_.clear();
    }
    cv_.notify_all();

    for (std::thread& worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    workers_.clear();
    reset();
}

void ChunkStreamingManager::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pendingBuilds_.clear();
    completedSubchunks_.clear();
    completedChunks_.clear();
    desiredChunks_.clear();
    loadedChunks_.clear();
    queuedChunks_.clear();
    rebuildingChunks_.clear();
    queuedGenerations_.clear();
    pendingChunkMeshes_.clear();
    loadedCenterChunkX_ = std::numeric_limits<int>::min();
    loadedCenterChunkZ_ = std::numeric_limits<int>::min();
}

ChunkLoadUpdateStats ChunkStreamingManager::updateLoadedChunks(
    int centerChunkX,
    int centerChunkZ,
    const std::function<void(ChunkCoord)>& unloadChunk)
{
    ChunkLoadUpdateStats stats{};
    std::vector<ChunkCoord> chunksToUnload;
    bool queuedAny = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        priorityCenterChunkX_ = centerChunkX;
        priorityCenterChunkZ_ = centerChunkZ;

        if (centerChunkX == loadedCenterChunkX_ && centerChunkZ == loadedCenterChunkZ_)
        {
            return stats;
        }

        const std::size_t sideLength = static_cast<std::size_t>(loadRadius_ * 2 + 1);
        desiredChunks_.clear();
        desiredChunks_.reserve(sideLength * sideLength);
        for (int dz = -loadRadius_; dz <= loadRadius_; ++dz)
        {
            for (int dx = -loadRadius_; dx <= loadRadius_; ++dx)
            {
                desiredChunks_.insert({centerChunkX + dx, centerChunkZ + dz});
            }
        }

        for (ChunkCoord coord : loadedChunks_)
        {
            if (!desiredChunks_.contains(coord))
            {
                chunksToUnload.push_back(coord);
            }
        }
        for (ChunkCoord coord : chunksToUnload)
        {
            loadedChunks_.erase(coord);
            rebuildingChunks_.erase(coord);
        }
        stats.unloaded = chunksToUnload.size();

        cancelQueuedChunksOutsideDesiredLocked();
        reprioritizePendingChunkBuildsLocked(centerChunkX, centerChunkZ);
        for (ChunkCoord coord : desiredChunks_)
        {
            if (loadedChunks_.contains(coord) || queuedChunks_.contains(coord))
            {
                continue;
            }
            queueChunkBuildLocked(coord, centerChunkX, centerChunkZ);
            ++stats.queued;
            queuedAny = true;
        }

        loadedCenterChunkX_ = centerChunkX;
        loadedCenterChunkZ_ = centerChunkZ;
    }

    for (ChunkCoord coord : chunksToUnload)
    {
        unloadChunk(coord);
    }
    if (queuedAny)
    {
        cv_.notify_all();
    }

    return stats;
}

std::size_t ChunkStreamingManager::processCompletedSubchunkBuilds()
{
    std::deque<SubchunkBuildResult> completedSubchunks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        completedSubchunks.swap(completedSubchunks_);
    }

    std::size_t processedCount = 0;
    while (!completedSubchunks.empty())
    {
        SubchunkBuildResult result = std::move(completedSubchunks.front());
        completedSubchunks.pop_front();
        ++processedCount;
        if (result.subchunkY < 0 || result.subchunkY >= kSubchunksPerChunk)
        {
            continue;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        const auto generationIt = queuedGenerations_.find(result.coord);
        if (generationIt == queuedGenerations_.end() || generationIt->second != result.generation)
        {
            continue;
        }
        if (!isChunkInCurrentLoadRangeLocked(result.coord) || loadedChunks_.contains(result.coord))
        {
            continue;
        }

        auto [pendingIt, inserted] = pendingChunkMeshes_.try_emplace(result.coord);
        PendingChunkMesh& pendingMesh = pendingIt->second;
        if (inserted)
        {
            pendingMesh.generation = result.generation;
        }
        if (pendingMesh.generation != result.generation)
        {
            continue;
        }

        const std::size_t subchunkIndex = static_cast<std::size_t>(result.subchunkY);
        if (pendingMesh.received[subchunkIndex])
        {
            continue;
        }

        pendingMesh.vertices[subchunkIndex] = std::move(result.vertices);
        pendingMesh.indices[subchunkIndex] = std::move(result.indices);
        pendingMesh.received[subchunkIndex] = true;
        ++pendingMesh.completedSubchunks;

        if (pendingMesh.completedSubchunks == kSubchunksPerChunk)
        {
            completedChunks_.push_back(assembleChunkMesh(result.coord, pendingMesh));
            pendingChunkMeshes_.erase(pendingIt);
        }
    }
    return processedCount;
}

bool ChunkStreamingManager::popCompletedChunkBuild(ChunkBuildResult& result)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (completedChunks_.empty())
    {
        return false;
    }

    result = std::move(completedChunks_.front());
    completedChunks_.pop_front();
    queuedChunks_.erase(result.coord);
    queuedGenerations_.erase(result.coord);
    pendingChunkMeshes_.erase(result.coord);
    return true;
}

bool ChunkStreamingManager::shouldAcceptCompletedChunk(ChunkCoord coord) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return isChunkInCurrentLoadRangeLocked(coord) &&
           (!loadedChunks_.contains(coord) || rebuildingChunks_.contains(coord));
}

void ChunkStreamingManager::markChunkLoaded(ChunkCoord coord)
{
    std::lock_guard<std::mutex> lock(mutex_);
    loadedChunks_.insert(coord);
    rebuildingChunks_.erase(coord);
}

bool ChunkStreamingManager::isChunkLoaded(ChunkCoord coord) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return loadedChunks_.contains(coord);
}

bool ChunkStreamingManager::rebuildLoadedChunk(ChunkCoord coord)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loadedChunks_.contains(coord))
        {
            return false;
        }

        queuedChunks_.erase(coord);
        queuedGenerations_.erase(coord);
        pendingChunkMeshes_.erase(coord);
        rebuildingChunks_.insert(coord);
        queueChunkBuildLocked(coord, priorityCenterChunkX_, priorityCenterChunkZ_);
    }

    cv_.notify_all();
    return true;
}

std::size_t ChunkStreamingManager::loadedChunkCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return loadedChunks_.size();
}

std::pair<std::size_t, std::size_t> ChunkStreamingManager::queueSizes() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return {pendingBuilds_.size(), completedSubchunks_.size() + completedChunks_.size()};
}

void ChunkStreamingManager::workerLoop()
{
    while (true)
    {
        ChunkBuildRequest request{};
        bool hasRequest = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]()
            {
                return !workerRunning_ || !pendingBuilds_.empty();
            });

            while (workerRunning_ && !pendingBuilds_.empty())
            {
                std::pop_heap(
                    pendingBuilds_.begin(),
                    pendingBuilds_.end(),
                    ChunkBuildRequestPriority{});
                request = pendingBuilds_.back();
                pendingBuilds_.pop_back();

                const auto generationIt = queuedGenerations_.find(request.coord);
                if (generationIt != queuedGenerations_.end() &&
                    generationIt->second == request.generation)
                {
                    const std::int64_t currentDistanceSq = chunkDistanceSq(
                        request.coord,
                        priorityCenterChunkX_,
                        priorityCenterChunkZ_);
                    if (currentDistanceSq != request.priorityDistanceSq)
                    {
                        request.priorityDistanceSq = currentDistanceSq;
                        pendingBuilds_.push_back(request);
                        std::push_heap(
                            pendingBuilds_.begin(),
                            pendingBuilds_.end(),
                            ChunkBuildRequestPriority{});
                        continue;
                    }

                    hasRequest = true;
                    break;
                }
            }

            if (!workerRunning_)
            {
                return;
            }
        }

        if (!hasRequest)
        {
            continue;
        }

        SubchunkBuildResult result = chunkMesher_.buildSubchunkMesh(request);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            completedSubchunks_.push_back(std::move(result));
        }
    }
}

void ChunkStreamingManager::queueChunkBuildLocked(ChunkCoord coord, int centerChunkX, int centerChunkZ)
{
    queuedChunks_.insert(coord);
    const std::uint64_t generation = ++buildGeneration_;
    queuedGenerations_[coord] = generation;
    const std::int64_t priorityDistanceSq = chunkDistanceSq(coord, centerChunkX, centerChunkZ);
    for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
    {
        pendingBuilds_.push_back({
            coord,
            subchunkY,
            priorityDistanceSq,
            generation,
        });
        std::push_heap(
            pendingBuilds_.begin(),
            pendingBuilds_.end(),
            ChunkBuildRequestPriority{});
    }
}

void ChunkStreamingManager::reprioritizePendingChunkBuildsLocked(int centerChunkX, int centerChunkZ)
{
    pendingBuilds_.erase(
        std::remove_if(
            pendingBuilds_.begin(),
            pendingBuilds_.end(),
            [this](const ChunkBuildRequest& request)
            {
                const auto generationIt = queuedGenerations_.find(request.coord);
                return generationIt == queuedGenerations_.end() ||
                       generationIt->second != request.generation;
            }),
        pendingBuilds_.end());
    for (ChunkBuildRequest& request : pendingBuilds_)
    {
        request.priorityDistanceSq = chunkDistanceSq(request.coord, centerChunkX, centerChunkZ);
    }
    std::make_heap(
        pendingBuilds_.begin(),
        pendingBuilds_.end(),
        ChunkBuildRequestPriority{});
}

std::size_t ChunkStreamingManager::cancelQueuedChunksOutsideDesiredLocked()
{
    std::size_t canceledCount = 0;
    for (auto it = queuedGenerations_.begin(); it != queuedGenerations_.end();)
    {
        if (desiredChunks_.contains(it->first))
        {
            ++it;
            continue;
        }

        queuedChunks_.erase(it->first);
        rebuildingChunks_.erase(it->first);
        pendingChunkMeshes_.erase(it->first);
        it = queuedGenerations_.erase(it);
        ++canceledCount;
    }
    return canceledCount;
}

bool ChunkStreamingManager::isChunkInLoadRange(ChunkCoord coord, int centerChunkX, int centerChunkZ) const
{
    return std::abs(coord.x - centerChunkX) <= loadRadius_ &&
           std::abs(coord.z - centerChunkZ) <= loadRadius_;
}

bool ChunkStreamingManager::isChunkInCurrentLoadRangeLocked(ChunkCoord coord) const
{
    if (loadedCenterChunkX_ == std::numeric_limits<int>::min() ||
        loadedCenterChunkZ_ == std::numeric_limits<int>::min())
    {
        return false;
    }
    return isChunkInLoadRange(coord, loadedCenterChunkX_, loadedCenterChunkZ_);
}

ChunkBuildResult ChunkStreamingManager::assembleChunkMesh(ChunkCoord coord, PendingChunkMesh& pendingMesh) const
{
    ChunkBuildResult result{};
    result.coord = coord;

    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
    {
        vertexCount += pendingMesh.vertices[static_cast<std::size_t>(subchunkY)].size();
        indexCount += pendingMesh.indices[static_cast<std::size_t>(subchunkY)].size();
    }

    result.vertices.reserve(vertexCount);
    result.indices.reserve(indexCount);
    result.subchunks.reserve(kSubchunksPerChunk);

    for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
    {
        auto& sourceVertices = pendingMesh.vertices[static_cast<std::size_t>(subchunkY)];
        auto& sourceIndices = pendingMesh.indices[static_cast<std::size_t>(subchunkY)];
        if (sourceIndices.empty())
        {
            continue;
        }

        SubchunkDraw draw{};
        draw.chunkX = coord.x;
        draw.chunkZ = coord.z;
        draw.subchunkY = subchunkY;
        draw.range.vertexCount = static_cast<std::uint32_t>(sourceVertices.size());
        draw.range.firstIndex = static_cast<std::uint32_t>(result.indices.size());
        draw.range.indexCount = static_cast<std::uint32_t>(sourceIndices.size());
        draw.range.vertexOffset = static_cast<std::int32_t>(result.vertices.size());

        result.vertices.insert(result.vertices.end(), sourceVertices.begin(), sourceVertices.end());
        result.indices.insert(result.indices.end(), sourceIndices.begin(), sourceIndices.end());
        result.subchunks.push_back(draw);
    }

    return result;
}
