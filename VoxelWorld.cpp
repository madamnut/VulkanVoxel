#include "VoxelWorld.h"
#include "WorldGenerator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr float kFrustumPaddingDegrees = 8.0f;
constexpr int kFaceNeighborOffsets[6][3] = {
    {0, 1, 0},
    {0, -1, 0},
    {0, 0, -1},
    {0, 0, 1},
    {-1, 0, 0},
    {1, 0, 0},
};
constexpr char kLevelMagic[4] = {'V', 'L', 'V', '1'};
constexpr char kRegionMagic[4] = {'V', 'R', 'G', '1'};
constexpr std::uint32_t kLevelVersion = 1;
constexpr std::uint8_t kStoredSubChunkUniform = 1;
constexpr std::uint8_t kStoredSubChunkDense = 2;
constexpr std::size_t kDirtySubChunkBuildBudgetPerPass = 12;
constexpr int kMeshBuildSampleSize = kSubChunkSize + 2;

struct RegionChunkIndexEntry {
    std::int32_t chunkX = 0;
    std::int32_t chunkZ = 0;
    std::uint64_t offset = 0;
    std::uint32_t size = 0;
};

std::size_t GetMeshBuildSampleIndex(int x, int y, int z) {
    return static_cast<std::size_t>(y * kMeshBuildSampleSize * kMeshBuildSampleSize +
                                    z * kMeshBuildSampleSize + x);
}

struct ClipVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

ClipVertex TransformPoint(const Mat4& matrix, const Vec3& point) {
    ClipVertex out{};
    out.x = matrix.m[0] * point.x + matrix.m[4] * point.y + matrix.m[8] * point.z + matrix.m[12];
    out.y = matrix.m[1] * point.x + matrix.m[5] * point.y + matrix.m[9] * point.z + matrix.m[13];
    out.z = matrix.m[2] * point.x + matrix.m[6] * point.y + matrix.m[10] * point.z + matrix.m[14];
    out.w = matrix.m[3] * point.x + matrix.m[7] * point.y + matrix.m[11] * point.z + matrix.m[15];
    return out;
}

template <typename MeshType>
void AppendQuad(
    MeshType& mesh,
    const Vec3& p0,
    const Vec3& p1,
    const Vec3& p2,
    const Vec3& p3,
    float uMax,
    float vMax
) {
    const std::uint32_t baseIndex = static_cast<std::uint32_t>(mesh.vertices.size());

    mesh.vertices.push_back({{p0.x, p0.y, p0.z}, {0.0f, 0.0f}});
    mesh.vertices.push_back({{p1.x, p1.y, p1.z}, {uMax, 0.0f}});
    mesh.vertices.push_back({{p2.x, p2.y, p2.z}, {uMax, vMax}});
    mesh.vertices.push_back({{p3.x, p3.y, p3.z}, {0.0f, vMax}});

    mesh.indices.push_back(baseIndex + 0);
    mesh.indices.push_back(baseIndex + 1);
    mesh.indices.push_back(baseIndex + 2);
    mesh.indices.push_back(baseIndex + 0);
    mesh.indices.push_back(baseIndex + 2);
    mesh.indices.push_back(baseIndex + 3);
}

template <typename Stream, typename T>
void WriteBinary(Stream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename Stream, typename T>
void ReadBinary(Stream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
void AppendBinary(std::vector<std::uint8_t>& buffer, const T& value) {
    const std::size_t offset = buffer.size();
    buffer.resize(offset + sizeof(T));
    std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

bool ShouldStoreSubChunk(const SubChunkVoxelData& subChunk) {
    return !subChunk.isUniform || subChunk.uniformBlock != 0;
}

}  // namespace

VoxelWorld::VoxelWorld() = default;
VoxelWorld::~VoxelWorld() = default;
VoxelWorld::VoxelWorld(VoxelWorld&&) noexcept = default;
VoxelWorld& VoxelWorld::operator=(VoxelWorld&&) noexcept = default;

std::int64_t VoxelWorld::MakeChunkKey(int chunkX, int chunkZ) {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(chunkX)) << 32) |
        static_cast<std::uint32_t>(chunkZ);
    return static_cast<std::int64_t>(key);
}

int VoxelWorld::FloorDiv(int value, int divisor) {
    const int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        return quotient - 1;
    }
    return quotient;
}

int VoxelWorld::PositiveMod(int value, int divisor) {
    const int remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

int VoxelWorld::GetRegionCoord(int chunkCoord) {
    return FloorDiv(chunkCoord, kRegionSizeInChunks);
}

int VoxelWorld::GetRegionLocalCoord(int chunkCoord) {
    return PositiveMod(chunkCoord, kRegionSizeInChunks);
}

std::int64_t VoxelWorld::MakeRegionKey(int regionX, int regionZ) {
    return MakeChunkKey(regionX, regionZ);
}

int VoxelWorld::GetSubChunkIndex(int worldY) {
    return worldY / kSubChunkSize;
}

int VoxelWorld::GetSubChunkBlockIndex(int localX, int localY, int localZ) {
    return localY * kSubChunkSize * kSubChunkSize + localZ * kSubChunkSize + localX;
}

std::uint16_t VoxelWorld::GetSubChunkBlock(const SubChunkVoxelData& subChunk, int localX, int localY, int localZ) {
    if (subChunk.isUniform) {
        return subChunk.uniformBlock;
    }

    return subChunk.blocks[static_cast<std::size_t>(GetSubChunkBlockIndex(localX, localY, localZ))];
}

void VoxelWorld::SetSubChunkBlock(SubChunkVoxelData& subChunk, int localX, int localY, int localZ, std::uint16_t blockValue) {
    if (subChunk.isUniform) {
        if (subChunk.uniformBlock == blockValue) {
            return;
        }

        subChunk.blocks.assign(static_cast<std::size_t>(kSubChunkVoxelCount), subChunk.uniformBlock);
        subChunk.isUniform = false;
    }

    subChunk.blocks[static_cast<std::size_t>(GetSubChunkBlockIndex(localX, localY, localZ))] = blockValue;
}

void VoxelWorld::TryCollapseSubChunk(SubChunkVoxelData& subChunk) {
    if (subChunk.isUniform || subChunk.blocks.empty()) {
        return;
    }

    const std::uint16_t candidate = subChunk.blocks.front();
    for (std::uint16_t blockValue : subChunk.blocks) {
        if (blockValue != candidate) {
            return;
        }
    }

    subChunk.blocks.clear();
    subChunk.blocks.shrink_to_fit();
    subChunk.uniformBlock = candidate;
    subChunk.isUniform = true;
}

std::uint16_t VoxelWorld::SampleGeneratedBlock(int worldY) const {
    return worldY < terrainConfig_.flatGroundHeight ? static_cast<std::uint16_t>(1) : static_cast<std::uint16_t>(0);
}

bool VoxelWorld::IsInsideWorld(int worldX, int worldY, int worldZ) {
    (void)worldX;
    (void)worldZ;
    return worldY >= 0 && worldY < kWorldSizeY;
}

std::string VoxelWorld::GetSaveRootPath() {
    return (std::filesystem::path(ASSET_DIR) / "saves" / "world").string();
}

std::string VoxelWorld::GetLevelFilePath() {
    return (std::filesystem::path(GetSaveRootPath()) / "level.dat").string();
}

std::string VoxelWorld::GetRegionDirectoryPath() {
    return (std::filesystem::path(GetSaveRootPath()) / "region").string();
}

std::string VoxelWorld::GetRegionFilePath(int regionX, int regionZ) {
    return (std::filesystem::path(GetRegionDirectoryPath()) /
            ("r." + std::to_string(regionX) + "." + std::to_string(regionZ) + ".vxr"))
        .string();
}

void VoxelWorld::UpdateStreamingTargets(int centerChunkX, int centerChunkZ, int keepRadius) {
    EnsureInitialized();
    RefreshStreamingQueue(centerChunkX, centerChunkZ, keepRadius);
}

std::vector<PendingChunkId> VoxelWorld::AcquireChunkLoadRequests(std::size_t maxCount) {
    EnsureInitialized();

    std::vector<PendingChunkId> requests;
    requests.reserve(std::min(maxCount, pendingChunkLoadQueue_.size()));

    while (!pendingChunkLoadQueue_.empty() && requests.size() < maxCount) {
        const PendingChunkId pendingChunk = pendingChunkLoadQueue_.front();
        pendingChunkLoadQueue_.pop_front();
        queuedChunkLoads_.erase(pendingChunk);

        if (chunkColumns_.contains(MakeChunkKey(pendingChunk.chunkX, pendingChunk.chunkZ))) {
            continue;
        }
        if (inFlightChunkLoads_.contains(pendingChunk)) {
            continue;
        }

        inFlightChunkLoads_.insert(pendingChunk);
        requests.push_back(pendingChunk);
    }

    return requests;
}

PreparedChunkColumn VoxelWorld::PrepareChunkColumn(int chunkX, int chunkZ) const {
    PreparedChunkColumn prepared{};
    prepared.id = {chunkX, chunkZ};

    const bool loadedFromDisk = LoadChunkColumn(chunkX, chunkZ, prepared.column);
    if (!loadedFromDisk) {
        GenerateChunkColumn(chunkX, chunkZ, prepared.column);
        prepared.column.modified = true;
        prepared.generated = true;
    }

    return prepared;
}

bool VoxelWorld::CommitPreparedChunkColumn(PreparedChunkColumn&& prepared) {
    EnsureInitialized();

    inFlightChunkLoads_.erase(prepared.id);

    const std::int64_t chunkKey = MakeChunkKey(prepared.id.chunkX, prepared.id.chunkZ);
    if (chunkColumns_.contains(chunkKey)) {
        return false;
    }

    if (prepared.generated) {
        saveDirty_ = true;
    }

    auto [it, inserted] = chunkColumns_.emplace(chunkKey, std::move(prepared.column));
    (void)inserted;

    if (desiredChunks_.contains(prepared.id)) {
        retiringChunks_.erase(prepared.id);
    } else {
        retiringChunks_.insert(prepared.id);
    }

    renderStatsDirty_ = true;
    EnqueueAllDirtySubChunks(prepared.id.chunkX, prepared.id.chunkZ, it->second);
    return true;
}

std::vector<MeshBuildInput> VoxelWorld::AcquireDirtyMeshRequests(std::size_t maxCount, int centerChunkX, int centerChunkZ, int keepRadius) {
    EnsureInitialized();

    std::vector<MeshBuildInput> requests;
    requests.reserve(maxCount);

    const std::size_t initialQueueCount = dirtySubChunkQueue_.size();
    for (std::size_t iteration = 0; iteration < initialQueueCount && requests.size() < maxCount; ++iteration) {
        if (dirtySubChunkQueue_.empty()) {
            break;
        }

        const DirtySubChunkId id = dirtySubChunkQueue_.front();
        dirtySubChunkQueue_.pop_front();
        queuedDirtySubChunks_.erase(id);

        const int dx = id.chunkX - centerChunkX;
        const int dz = id.chunkZ - centerChunkZ;
        if (dx * dx + dz * dz > keepRadius * keepRadius) {
            EnqueueDirtySubChunk(id.chunkX, id.chunkZ, id.subChunkIndex);
            continue;
        }

        if (inFlightDirtySubChunks_.contains(id)) {
            continue;
        }

        const auto columnIt = chunkColumns_.find(MakeChunkKey(id.chunkX, id.chunkZ));
        if (columnIt == chunkColumns_.end()) {
            continue;
        }
        if (columnIt->second.subChunkMeshes.size() != kSubChunkCountY ||
            columnIt->second.subChunks.size() != kSubChunkCountY) {
            continue;
        }
        if (!columnIt->second.subChunkMeshes[static_cast<std::size_t>(id.subChunkIndex)].dirty) {
            continue;
        }

        MeshBuildInput input{};
        input.id = id;
        input.minWorldX = id.chunkX * kChunkSizeX;
        input.minWorldY = id.subChunkIndex * kSubChunkSize;
        input.minWorldZ = id.chunkZ * kChunkSizeZ;
        input.blocks.resize(static_cast<std::size_t>(kMeshBuildSampleSize * kMeshBuildSampleSize * kMeshBuildSampleSize));

        for (int sampleY = -1; sampleY <= kSubChunkSize; ++sampleY) {
            for (int sampleZ = -1; sampleZ <= kSubChunkSize; ++sampleZ) {
                for (int sampleX = -1; sampleX <= kSubChunkSize; ++sampleX) {
                    input.blocks[GetMeshBuildSampleIndex(sampleX + 1, sampleY + 1, sampleZ + 1)] =
                        GetBlock(input.minWorldX + sampleX, input.minWorldY + sampleY, input.minWorldZ + sampleZ);
                }
            }
        }

        inFlightDirtySubChunks_.insert(id);
        requests.push_back(std::move(input));
    }

    return requests;
}

PreparedSubChunkMesh VoxelWorld::PrepareSubChunkMesh(const MeshBuildInput& input) const {
    PreparedSubChunkMesh preparedMesh{};
    preparedMesh.id = input.id;

    constexpr int dims[3] = {kChunkSizeX, kSubChunkSize, kChunkSizeZ};
    std::vector<int> mask(static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ));

    const auto sampleBlock = [&input](int localX, int localY, int localZ) -> std::uint16_t {
        return input.blocks[GetMeshBuildSampleIndex(localX + 1, localY + 1, localZ + 1)];
    };

    if (sampleBlock(0, 0, 0) == 0) {
        bool hasAnySolid = false;
        for (std::uint16_t blockValue : input.blocks) {
            if (blockValue != 0) {
                hasAnySolid = true;
                break;
            }
        }
        if (!hasAnySolid) {
            return preparedMesh;
        }
    }

    const auto emitGreedyQuad = [&](int axis, bool positiveNormal, int slice, int startU, int startV, int width, int height) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;

        Vec3 base{};
        if (axis == 0) {
            base.x = static_cast<float>(input.minWorldX + slice);
        } else if (axis == 1) {
            base.y = static_cast<float>(input.minWorldY + slice);
        } else {
            base.z = static_cast<float>(input.minWorldZ + slice);
        }

        if (u == 0) {
            base.x = static_cast<float>(input.minWorldX + startU);
        } else if (u == 1) {
            base.y = static_cast<float>(input.minWorldY + startU);
        } else {
            base.z = static_cast<float>(input.minWorldZ + startU);
        }

        if (v == 0) {
            base.x = static_cast<float>(input.minWorldX + startV);
        } else if (v == 1) {
            base.y = static_cast<float>(input.minWorldY + startV);
        } else {
            base.z = static_cast<float>(input.minWorldZ + startV);
        }

        Vec3 du{};
        Vec3 dv{};
        if (u == 0) {
            du.x = static_cast<float>(width);
        } else if (u == 1) {
            du.y = static_cast<float>(width);
        } else {
            du.z = static_cast<float>(width);
        }

        if (v == 0) {
            dv.x = static_cast<float>(height);
        } else if (v == 1) {
            dv.y = static_cast<float>(height);
        } else {
            dv.z = static_cast<float>(height);
        }

        const Vec3 p0 = base;
        const Vec3 p1 = positiveNormal ? base + du : base + dv;
        const Vec3 p2 = base + du + dv;
        const Vec3 p3 = positiveNormal ? base + dv : base + du;
        AppendQuad(preparedMesh, p0, p1, p2, p3, static_cast<float>(width), static_cast<float>(height));
    };

    for (int axis = 0; axis < 3; ++axis) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;

        for (int slice = -1; slice < dims[axis]; ++slice) {
            int maskIndex = 0;
            for (int j = 0; j < dims[v]; ++j) {
                for (int i = 0; i < dims[u]; ++i) {
                    int aCoords[3] = {};
                    int bCoords[3] = {};
                    aCoords[axis] = slice;
                    bCoords[axis] = slice + 1;
                    aCoords[u] = i;
                    bCoords[u] = i;
                    aCoords[v] = j;
                    bCoords[v] = j;

                    const std::uint16_t a = sampleBlock(aCoords[0], aCoords[1], aCoords[2]);
                    const std::uint16_t b = sampleBlock(bCoords[0], bCoords[1], bCoords[2]);

                    if (a != 0 && b == 0) {
                        mask[static_cast<std::size_t>(maskIndex)] = static_cast<int>(a);
                    } else if (a == 0 && b != 0) {
                        mask[static_cast<std::size_t>(maskIndex)] = -static_cast<int>(b);
                    } else {
                        mask[static_cast<std::size_t>(maskIndex)] = 0;
                    }

                    ++maskIndex;
                }
            }

            for (int j = 0; j < dims[v]; ++j) {
                for (int i = 0; i < dims[u];) {
                    const int current = mask[static_cast<std::size_t>(i + j * dims[u])];
                    if (current == 0) {
                        ++i;
                        continue;
                    }

                    int width = 1;
                    while (i + width < dims[u] &&
                           mask[static_cast<std::size_t>(i + width + j * dims[u])] == current) {
                        ++width;
                    }

                    int height = 1;
                    bool done = false;
                    while (j + height < dims[v] && !done) {
                        for (int k = 0; k < width; ++k) {
                            if (mask[static_cast<std::size_t>(i + k + (j + height) * dims[u])] != current) {
                                done = true;
                                break;
                            }
                        }
                        if (!done) {
                            ++height;
                        }
                    }

                    emitGreedyQuad(axis, current > 0, slice + 1, i, j, width, height);

                    for (int row = 0; row < height; ++row) {
                        for (int col = 0; col < width; ++col) {
                            mask[static_cast<std::size_t>(i + col + (j + row) * dims[u])] = 0;
                        }
                    }

                    i += width;
                }
            }
        }
    }

    return preparedMesh;
}

bool VoxelWorld::CommitPreparedSubChunkMesh(PreparedSubChunkMesh&& preparedMesh) {
    const DirtySubChunkId id = preparedMesh.id;
    inFlightDirtySubChunks_.erase(id);

    const auto columnIt = chunkColumns_.find(MakeChunkKey(id.chunkX, id.chunkZ));
    if (columnIt == chunkColumns_.end()) {
        return false;
    }
    if (id.subChunkIndex < 0 || id.subChunkIndex >= kSubChunkCountY) {
        return false;
    }
    if (columnIt->second.subChunkMeshes.size() != kSubChunkCountY) {
        return false;
    }

    SubChunkMeshData& mesh = columnIt->second.subChunkMeshes[static_cast<std::size_t>(id.subChunkIndex)];
    mesh.vertices = std::move(preparedMesh.vertices);
    mesh.indices = std::move(preparedMesh.indices);
    mesh.dirty = queuedDirtySubChunks_.contains(id);
    ++mesh.revision;
    QueueRenderChunkUpdate(id.chunkX, id.chunkZ);
    return true;
}

WorldRenderUpdate VoxelWorld::DrainRenderUpdates() {
    EnsureInitialized();

    WorldRenderUpdate update{};
    update.loadedChunkCount = chunkColumns_.size();

    update.removals.reserve(pendingRenderChunkRemovals_.size());
    for (const PendingChunkId& chunkId : pendingRenderChunkRemovals_) {
        update.removals.push_back(chunkId);
    }
    pendingRenderChunkRemovals_.clear();

    update.uploads.reserve(pendingRenderChunkUpdates_.size());
    for (const PendingChunkId& chunkId : pendingRenderChunkUpdates_) {
        update.uploads.push_back(chunkId);
    }
    pendingRenderChunkUpdates_.clear();
    renderStatsDirty_ = false;

    return update;
}

bool VoxelWorld::HasPendingRenderUpdates() const {
    return renderStatsDirty_ || !pendingRenderChunkUpdates_.empty() || !pendingRenderChunkRemovals_.empty();
}

std::size_t VoxelWorld::FinalizeStreamingWindow(int centerChunkX, int centerChunkZ, int keepRadius, std::size_t unloadBudget) {
    const std::size_t remainingMissingCount = pendingChunkLoadQueue_.size() + inFlightChunkLoads_.size();
    const std::size_t remainingDirtyCount = dirtySubChunkQueue_.size() + inFlightDirtySubChunks_.size();
    UnloadRetiredChunks(unloadBudget);
    const std::size_t remainingRetiringCount = retiringChunks_.size();
    return remainingMissingCount + remainingDirtyCount + remainingRetiringCount;
}

std::size_t VoxelWorld::UpdateStreamingWindow(int centerChunkX, int centerChunkZ, int keepRadius, std::size_t generationBudget) {
    UpdateStreamingTargets(centerChunkX, centerChunkZ, keepRadius);

    const std::vector<PendingChunkId> requests = AcquireChunkLoadRequests(generationBudget);
    for (const PendingChunkId& request : requests) {
        CommitPreparedChunkColumn(PrepareChunkColumn(request.chunkX, request.chunkZ));
    }

    const std::vector<MeshBuildInput> meshRequests =
        AcquireDirtyMeshRequests(kDirtySubChunkBuildBudgetPerPass, centerChunkX, centerChunkZ, keepRadius);
    for (const MeshBuildInput& meshRequest : meshRequests) {
        CommitPreparedSubChunkMesh(PrepareSubChunkMesh(meshRequest));
    }

    return FinalizeStreamingWindow(centerChunkX, centerChunkZ, keepRadius, generationBudget);
}

void VoxelWorld::EnsureChunkColumn(int chunkX, int chunkZ) {
    EnsureInitialized();

    const std::int64_t chunkKey = MakeChunkKey(chunkX, chunkZ);
    if (chunkColumns_.contains(chunkKey)) {
        return;
    }

    ChunkColumnData column{};
    const bool loadedFromDisk = LoadChunkColumn(chunkX, chunkZ, column);
    if (!loadedFromDisk) {
        GenerateChunkColumn(chunkX, chunkZ, column);
        column.modified = true;
        saveDirty_ = true;
    }

    auto [it, inserted] = chunkColumns_.emplace(chunkKey, std::move(column));
    (void)inserted;

    const PendingChunkId id{chunkX, chunkZ};
    if (desiredChunks_.contains(id)) {
        retiringChunks_.erase(id);
    } else {
        retiringChunks_.insert(id);
    }

    EnqueueAllDirtySubChunks(chunkX, chunkZ, it->second);
}

void VoxelWorld::EnsureRange(int minChunkX, int maxChunkX, int minChunkZ, int maxChunkZ) {
    if (minChunkX > maxChunkX || minChunkZ > maxChunkZ) {
        return;
    }

    for (int chunkZ = minChunkZ; chunkZ <= maxChunkZ; ++chunkZ) {
        for (int chunkX = minChunkX; chunkX <= maxChunkX; ++chunkX) {
            EnsureChunkColumn(chunkX, chunkZ);
        }
    }
}

void VoxelWorld::EnsureInitialized() {
    if (initialized_) {
        return;
    }

    EnsureTerrainConfigLoaded();
    LoadOrCreateSave();
    initialized_ = true;
}

void VoxelWorld::InitializeVoxelSubChunks(ChunkColumnData& column) const {
    column.subChunks.clear();
    column.subChunks.resize(kSubChunkCountY);
    for (SubChunkVoxelData& subChunk : column.subChunks) {
        subChunk.blocks.clear();
        subChunk.uniformBlock = 0;
        subChunk.isUniform = true;
    }
}

void VoxelWorld::InitializeSubChunkMeshes(ChunkColumnData& column, bool dirty) const {
    column.subChunkMeshes.clear();
    column.subChunkMeshes.resize(kSubChunkCountY);
    for (SubChunkMeshData& subChunkMesh : column.subChunkMeshes) {
        subChunkMesh.vertices.clear();
        subChunkMesh.indices.clear();
        subChunkMesh.dirty = dirty;
        subChunkMesh.revision = 0;
    }
}

std::unordered_set<PendingChunkId, PendingChunkIdHash> VoxelWorld::CollectDesiredChunks(int centerChunkX, int centerChunkZ, int keepRadius) const {
    std::unordered_set<PendingChunkId, PendingChunkIdHash> desiredChunks;
    desiredChunks.reserve(static_cast<std::size_t>((keepRadius * 2 + 1) * (keepRadius * 2 + 1)));

    for (int dz = -keepRadius; dz <= keepRadius; ++dz) {
        for (int dx = -keepRadius; dx <= keepRadius; ++dx) {
            if (dx * dx + dz * dz > keepRadius * keepRadius) {
                continue;
            }

            desiredChunks.insert(PendingChunkId{centerChunkX + dx, centerChunkZ + dz});
        }
    }

    return desiredChunks;
}

void VoxelWorld::RebuildPendingChunkQueue(int centerChunkX, int centerChunkZ) {
    struct PendingChunk {
        PendingChunkId id{};
        int distanceSquared = 0;
    };

    std::vector<PendingChunk> pendingChunks;
    pendingChunks.reserve(desiredChunks_.size());
    for (const PendingChunkId& id : desiredChunks_) {
        if (chunkColumns_.contains(MakeChunkKey(id.chunkX, id.chunkZ))) {
            continue;
        }

        const int dx = id.chunkX - centerChunkX;
        const int dz = id.chunkZ - centerChunkZ;
        pendingChunks.push_back({id, dx * dx + dz * dz});
    }

    std::sort(pendingChunks.begin(), pendingChunks.end(), [](const PendingChunk& lhs, const PendingChunk& rhs) {
        if (lhs.distanceSquared != rhs.distanceSquared) {
            return lhs.distanceSquared < rhs.distanceSquared;
        }
        if (lhs.id.chunkZ != rhs.id.chunkZ) {
            return lhs.id.chunkZ < rhs.id.chunkZ;
        }
        return lhs.id.chunkX < rhs.id.chunkX;
    });

    pendingChunkLoadQueue_.clear();
    queuedChunkLoads_.clear();
    for (const PendingChunk& pendingChunk : pendingChunks) {
        pendingChunkLoadQueue_.push_back(pendingChunk.id);
        queuedChunkLoads_.insert(pendingChunk.id);
    }
}

void VoxelWorld::RefreshStreamingQueue(int centerChunkX, int centerChunkZ, int keepRadius) {
    if (streamingWindowInitialized_ &&
        streamingCenterChunkX_ == centerChunkX &&
        streamingCenterChunkZ_ == centerChunkZ &&
        streamingKeepRadius_ == keepRadius) {
        return;
    }

    streamingWindowInitialized_ = true;
    streamingCenterChunkX_ = centerChunkX;
    streamingCenterChunkZ_ = centerChunkZ;
    streamingKeepRadius_ = keepRadius;

    const std::unordered_set<PendingChunkId, PendingChunkIdHash> previousDesiredChunks = desiredChunks_;
    const std::unordered_set<PendingChunkId, PendingChunkIdHash> newDesiredChunks =
        CollectDesiredChunks(centerChunkX, centerChunkZ, keepRadius);

    for (const PendingChunkId& id : previousDesiredChunks) {
        if (!newDesiredChunks.contains(id) && chunkColumns_.contains(MakeChunkKey(id.chunkX, id.chunkZ))) {
            retiringChunks_.insert(id);
            QueueRenderChunkRemoval(id.chunkX, id.chunkZ);
        }
    }

    for (const PendingChunkId& id : newDesiredChunks) {
        retiringChunks_.erase(id);
    }

    desiredChunks_ = newDesiredChunks;

    for (const PendingChunkId& id : desiredChunks_) {
        if (previousDesiredChunks.contains(id)) {
            continue;
        }

        const auto columnIt = chunkColumns_.find(MakeChunkKey(id.chunkX, id.chunkZ));
        if (columnIt != chunkColumns_.end()) {
            QueueRenderUpdatesForChunk(id.chunkX, id.chunkZ, columnIt->second);
        }
    }

    for (auto it = retiringChunks_.begin(); it != retiringChunks_.end();) {
        if (!chunkColumns_.contains(MakeChunkKey(it->chunkX, it->chunkZ)) || desiredChunks_.contains(*it)) {
            it = retiringChunks_.erase(it);
            continue;
        }

        ++it;
    }

    RebuildPendingChunkQueue(centerChunkX, centerChunkZ);
}

void VoxelWorld::EnqueueDirtySubChunk(int chunkX, int chunkZ, int subChunkIndex) {
    if (subChunkIndex < 0 || subChunkIndex >= kSubChunkCountY) {
        return;
    }

    const DirtySubChunkId id{chunkX, chunkZ, subChunkIndex};
    if (!queuedDirtySubChunks_.insert(id).second) {
        return;
    }

    dirtySubChunkQueue_.push_back(id);
}

void VoxelWorld::EnqueueAllDirtySubChunks(int chunkX, int chunkZ, ChunkColumnData& column) {
    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        for (int subChunkIndex = kSubChunkCountY - 1; subChunkIndex >= 0; --subChunkIndex) {
            EnqueueDirtySubChunk(chunkX, chunkZ, subChunkIndex);
        }
        return;
    }

    for (int subChunkIndex = kSubChunkCountY - 1; subChunkIndex >= 0; --subChunkIndex) {
        SubChunkMeshData& mesh = column.subChunkMeshes[static_cast<std::size_t>(subChunkIndex)];
        if (!mesh.dirty) {
            continue;
        }

        if (column.subChunks.size() == kSubChunkCountY) {
            const SubChunkVoxelData& voxelSubChunk = column.subChunks[static_cast<std::size_t>(subChunkIndex)];
            if (voxelSubChunk.isUniform && voxelSubChunk.uniformBlock == 0) {
                mesh.dirty = false;
                continue;
            }
        }

        EnqueueDirtySubChunk(chunkX, chunkZ, subChunkIndex);
    }
}

void VoxelWorld::RemoveQueuedDirtySubChunksForChunk(int chunkX, int chunkZ) {
    if (dirtySubChunkQueue_.empty()) {
        return;
    }

    std::deque<DirtySubChunkId> retainedQueue;
    while (!dirtySubChunkQueue_.empty()) {
        DirtySubChunkId id = dirtySubChunkQueue_.front();
        dirtySubChunkQueue_.pop_front();
        if (id.chunkX == chunkX && id.chunkZ == chunkZ) {
            queuedDirtySubChunks_.erase(id);
            continue;
        }

        retainedQueue.push_back(id);
    }

    dirtySubChunkQueue_ = std::move(retainedQueue);
}

void VoxelWorld::MarkSubChunkDirty(int chunkX, int chunkZ, int subChunkIndex) {
    if (subChunkIndex < 0 || subChunkIndex >= kSubChunkCountY) {
        return;
    }

    const auto columnIt = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (columnIt == chunkColumns_.end()) {
        return;
    }

    if (columnIt->second.subChunkMeshes.size() != kSubChunkCountY) {
        InitializeSubChunkMeshes(columnIt->second, true);
        EnqueueAllDirtySubChunks(chunkX, chunkZ, columnIt->second);
        return;
    }

    SubChunkMeshData& subChunkMesh = columnIt->second.subChunkMeshes[static_cast<std::size_t>(subChunkIndex)];
    subChunkMesh.dirty = true;
    EnqueueDirtySubChunk(chunkX, chunkZ, subChunkIndex);
}

bool VoxelWorld::IsChunkDesired(int chunkX, int chunkZ) const {
    return desiredChunks_.contains(PendingChunkId{chunkX, chunkZ});
}

void VoxelWorld::QueueRenderChunkUpdate(int chunkX, int chunkZ) {
    const PendingChunkId chunkId{chunkX, chunkZ};
    if (!IsChunkDesired(chunkX, chunkZ)) {
        return;
    }

    const auto columnIt = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (columnIt == chunkColumns_.end() || columnIt->second.subChunkMeshes.size() != kSubChunkCountY) {
        return;
    }

    bool hasRenderableMesh = false;
    for (const SubChunkMeshData& mesh : columnIt->second.subChunkMeshes) {
        if (!mesh.vertices.empty() && !mesh.indices.empty()) {
            hasRenderableMesh = true;
            break;
        }
    }

    pendingRenderChunkRemovals_.erase(chunkId);
    pendingRenderChunkUpdates_.erase(chunkId);

    if (!hasRenderableMesh) {
        if (activeRenderChunks_.erase(chunkId) > 0) {
            pendingRenderChunkRemovals_.insert(chunkId);
        }
        return;
    }

    activeRenderChunks_.insert(chunkId);
    pendingRenderChunkUpdates_.insert(chunkId);
}

void VoxelWorld::QueueRenderChunkRemoval(int chunkX, int chunkZ) {
    const PendingChunkId chunkId{chunkX, chunkZ};
    pendingRenderChunkUpdates_.erase(chunkId);
    if (activeRenderChunks_.erase(chunkId) > 0) {
        pendingRenderChunkRemovals_.insert(chunkId);
    }
}

void VoxelWorld::QueueRenderUpdatesForChunk(int chunkX, int chunkZ, const ChunkColumnData& column) {
    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        return;
    }

    QueueRenderChunkUpdate(chunkX, chunkZ);
}

std::size_t VoxelWorld::RebuildDirtyMeshesInWindow(int centerChunkX, int centerChunkZ, int keepRadius) {
    std::size_t rebuiltCount = 0;
    const std::size_t initialQueueCount = dirtySubChunkQueue_.size();

    for (std::size_t iteration = 0; iteration < initialQueueCount && rebuiltCount < kDirtySubChunkBuildBudgetPerPass; ++iteration) {
        if (dirtySubChunkQueue_.empty()) {
            break;
        }

        const DirtySubChunkId id = dirtySubChunkQueue_.front();
        dirtySubChunkQueue_.pop_front();
        queuedDirtySubChunks_.erase(id);

        const int dx = id.chunkX - centerChunkX;
        const int dz = id.chunkZ - centerChunkZ;
        if (dx * dx + dz * dz > keepRadius * keepRadius) {
            EnqueueDirtySubChunk(id.chunkX, id.chunkZ, id.subChunkIndex);
            continue;
        }

        const auto columnIt = chunkColumns_.find(MakeChunkKey(id.chunkX, id.chunkZ));
        if (columnIt == chunkColumns_.end()) {
            continue;
        }

        if (columnIt->second.subChunkMeshes.size() != kSubChunkCountY) {
            InitializeSubChunkMeshes(columnIt->second, true);
            EnqueueDirtySubChunk(id.chunkX, id.chunkZ, id.subChunkIndex);
            continue;
        }

        if (!columnIt->second.subChunkMeshes[static_cast<std::size_t>(id.subChunkIndex)].dirty) {
            continue;
        }

        RebuildSubChunkMesh(id.chunkX, id.chunkZ, id.subChunkIndex);
        ++rebuiltCount;
    }

    std::size_t remainingRelevantCount = 0;
    for (const DirtySubChunkId& id : dirtySubChunkQueue_) {
        const int dx = id.chunkX - centerChunkX;
        const int dz = id.chunkZ - centerChunkZ;
        if (dx * dx + dz * dz <= keepRadius * keepRadius) {
            ++remainingRelevantCount;
        }
    }

    return remainingRelevantCount;
}

void VoxelWorld::UnloadRetiredChunks(std::size_t unloadBudget) {
    struct PendingUnload {
        int chunkX = 0;
        int chunkZ = 0;
    };

    std::vector<PendingUnload> unloads;
    unloads.reserve(retiringChunks_.size());
    for (const PendingChunkId& id : retiringChunks_) {
        if (unloads.size() >= unloadBudget) {
            break;
        }

        if (desiredChunks_.contains(id)) {
            continue;
        }

        if (!chunkColumns_.contains(MakeChunkKey(id.chunkX, id.chunkZ))) {
            continue;
        }

        unloads.push_back({id.chunkX, id.chunkZ});
    }

    if (unloads.empty()) {
        return;
    }

    struct RegionUnloadGroup {
        int regionX = 0;
        int regionZ = 0;
        std::vector<PendingUnload> chunks;
    };

    std::unordered_map<std::int64_t, RegionUnloadGroup> unloadRegions;
    for (const PendingUnload& unload : unloads) {
        const int regionX = GetRegionCoord(unload.chunkX);
        const int regionZ = GetRegionCoord(unload.chunkZ);
        RegionUnloadGroup& group = unloadRegions[MakeRegionKey(regionX, regionZ)];
        group.regionX = regionX;
        group.regionZ = regionZ;
        group.chunks.push_back(unload);
    }

    for (auto& [regionKey, group] : unloadRegions) {
        (void)regionKey;
        bool regionModified = false;
        std::unordered_map<std::int64_t, const ChunkColumnData*> overrides;

        for (const PendingUnload& unload : group.chunks) {
            auto columnIt = chunkColumns_.find(MakeChunkKey(unload.chunkX, unload.chunkZ));
            if (columnIt == chunkColumns_.end()) {
                continue;
            }

            if (columnIt->second.modified) {
                regionModified = true;
                overrides[MakeChunkKey(unload.chunkX, unload.chunkZ)] = &columnIt->second;
            }
        }

        if (regionModified) {
            SaveRegionOverrides(group.regionX, group.regionZ, overrides);
        }

        for (const PendingUnload& unload : group.chunks) {
            auto columnIt = chunkColumns_.find(MakeChunkKey(unload.chunkX, unload.chunkZ));
            if (columnIt == chunkColumns_.end()) {
                continue;
            }

            if (columnIt->second.modified) {
                columnIt->second.modified = false;
            }
            retiringChunks_.erase(PendingChunkId{unload.chunkX, unload.chunkZ});
            QueueRenderChunkRemoval(unload.chunkX, unload.chunkZ);
            RemoveQueuedDirtySubChunksForChunk(unload.chunkX, unload.chunkZ);
            chunkColumns_.erase(columnIt);
            renderStatsDirty_ = true;
        }
    }

    saveDirty_ = false;
    for (const auto& [key, column] : chunkColumns_) {
        (void)key;
        if (column.modified) {
            saveDirty_ = true;
            break;
        }
    }
}

void VoxelWorld::GenerateChunkColumn(int chunkX, int chunkZ, ChunkColumnData& outColumn) const {
    WorldGenerator::GenerateChunkColumn(chunkX, chunkZ, terrainConfig_, outColumn);
}

void VoxelWorld::LoadOrCreateSave() {
    const std::filesystem::path levelPath = GetLevelFilePath();
    const std::filesystem::path regionDirectory = GetRegionDirectoryPath();

    std::filesystem::create_directories(regionDirectory);

    auto writeFreshLevel = [&]() {
        std::ofstream stream(levelPath, std::ios::binary | std::ios::trunc);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to write world level file.");
        }

        stream.write(kLevelMagic, sizeof(kLevelMagic));
        WriteBinary(stream, kLevelVersion);
        WriteBinary(stream, static_cast<std::int32_t>(kWorldSizeY));
        WriteBinary(stream, static_cast<std::int32_t>(kChunkSizeX));
        WriteBinary(stream, static_cast<std::int32_t>(kChunkSizeZ));
        WriteBinary(stream, static_cast<std::int32_t>(kSubChunkSize));
        WriteBinary(stream, static_cast<std::int32_t>(terrainConfig_.seed));
        if (!stream) {
            throw std::runtime_error("Failed while initializing level.dat.");
        }
    };

    if (!std::filesystem::exists(levelPath)) {
        writeFreshLevel();
        saveDirty_ = false;
        return;
    }

    std::ifstream stream(levelPath, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open world level file.");
    }

    char magic[4] = {};
    std::uint32_t version = 0;
    std::int32_t worldHeight = 0;
    std::int32_t chunkSizeX = 0;
    std::int32_t chunkSizeZ = 0;
    std::int32_t subChunkSize = 0;
    std::int32_t seed = 0;
    stream.read(magic, sizeof(magic));
    ReadBinary(stream, version);
    ReadBinary(stream, worldHeight);
    ReadBinary(stream, chunkSizeX);
    ReadBinary(stream, chunkSizeZ);
    ReadBinary(stream, subChunkSize);
    ReadBinary(stream, seed);

    const bool valid =
        stream &&
        std::memcmp(magic, kLevelMagic, sizeof(kLevelMagic)) == 0 &&
        version == kLevelVersion &&
        worldHeight == kWorldSizeY &&
        chunkSizeX == kChunkSizeX &&
        chunkSizeZ == kChunkSizeZ &&
        subChunkSize == kSubChunkSize;

    if (!valid) {
        std::filesystem::remove_all(regionDirectory);
        std::filesystem::create_directories(regionDirectory);
        writeFreshLevel();
    }

    saveDirty_ = false;
}

void VoxelWorld::SaveAllDirtyChunks() {
    struct DirtyChunkRef {
        int chunkX = 0;
        int chunkZ = 0;
    };

    struct DirtyRegionGroup {
        int regionX = 0;
        int regionZ = 0;
        std::vector<DirtyChunkRef> chunks;
    };

    std::unordered_map<std::int64_t, DirtyRegionGroup> dirtyRegions;
    for (const auto& [key, column] : chunkColumns_) {
        if (!column.modified) {
            continue;
        }

        const std::uint64_t unsignedKey = static_cast<std::uint64_t>(key);
        const int chunkX = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey >> 32));
        const int chunkZ = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey));
        const int regionX = GetRegionCoord(chunkX);
        const int regionZ = GetRegionCoord(chunkZ);
        DirtyRegionGroup& regionGroup = dirtyRegions[MakeRegionKey(regionX, regionZ)];
        regionGroup.regionX = regionX;
        regionGroup.regionZ = regionZ;
        regionGroup.chunks.push_back({chunkX, chunkZ});
    }

    for (auto& [regionKey, regionGroup] : dirtyRegions) {
        (void)regionKey;
        std::unordered_map<std::int64_t, const ChunkColumnData*> overrides;

        for (const DirtyChunkRef& dirtyChunk : regionGroup.chunks) {
            auto columnIt = chunkColumns_.find(MakeChunkKey(dirtyChunk.chunkX, dirtyChunk.chunkZ));
            if (columnIt == chunkColumns_.end()) {
                continue;
            }

            overrides[MakeChunkKey(dirtyChunk.chunkX, dirtyChunk.chunkZ)] = &columnIt->second;
            columnIt->second.modified = false;
        }

        SaveRegionOverrides(regionGroup.regionX, regionGroup.regionZ, overrides);
    }

    saveDirty_ = false;
}

bool VoxelWorld::LoadChunkColumn(int chunkX, int chunkZ, ChunkColumnData& outColumn) const {
    const std::filesystem::path regionPath = GetRegionFilePath(GetRegionCoord(chunkX), GetRegionCoord(chunkZ));
    if (!std::filesystem::exists(regionPath)) {
        return false;
    }

    std::ifstream stream(regionPath, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open region file.");
    }

    char magic[4] = {};
    std::int32_t fileRegionX = 0;
    std::int32_t fileRegionZ = 0;
    std::uint32_t chunkCount = 0;
    stream.read(magic, sizeof(magic));
    ReadBinary(stream, fileRegionX);
    ReadBinary(stream, fileRegionZ);
    ReadBinary(stream, chunkCount);

    const int regionX = GetRegionCoord(chunkX);
    const int regionZ = GetRegionCoord(chunkZ);
    if (!stream || std::memcmp(magic, kRegionMagic, sizeof(kRegionMagic)) != 0 ||
        fileRegionX != regionX || fileRegionZ != regionZ) {
        throw std::runtime_error("Region file is corrupted or incompatible.");
    }

    for (std::uint32_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
        RegionChunkIndexEntry indexEntry{};
        ReadBinary(stream, indexEntry.chunkX);
        ReadBinary(stream, indexEntry.chunkZ);
        ReadBinary(stream, indexEntry.offset);
        ReadBinary(stream, indexEntry.size);

        if (!stream) {
            throw std::runtime_error("Region index is corrupted.");
        }

        if (indexEntry.chunkX != chunkX || indexEntry.chunkZ != chunkZ) {
            continue;
        }

        stream.seekg(static_cast<std::streamoff>(indexEntry.offset), std::ios::beg);
        if (!stream) {
            throw std::runtime_error("Region chunk offset is invalid.");
        }

        DeserializeChunkColumnPayload(stream, outColumn);
        outColumn.modified = false;
        return true;
    }

    return false;
}

void VoxelWorld::SaveChunkColumn(int chunkX, int chunkZ, ChunkColumnData& column) {
    if (!column.modified) {
        return;
    }

    const int regionX = GetRegionCoord(chunkX);
    const int regionZ = GetRegionCoord(chunkZ);
    std::unordered_map<std::int64_t, const ChunkColumnData*> overrides;
    overrides.emplace(MakeChunkKey(chunkX, chunkZ), &column);
    SaveRegionOverrides(regionX, regionZ, overrides);
    column.modified = false;
}

std::vector<std::uint8_t> VoxelWorld::SerializeChunkColumnPayload(const ChunkColumnData& column) const {
    std::vector<std::uint8_t> payload;

    std::uint32_t storedSubChunkCount = 0;
    for (const SubChunkVoxelData& subChunk : column.subChunks) {
        if (ShouldStoreSubChunk(subChunk)) {
            ++storedSubChunkCount;
        }
    }
    AppendBinary(payload, storedSubChunkCount);

    for (std::size_t subChunkIndex = 0; subChunkIndex < column.subChunks.size(); ++subChunkIndex) {
        const SubChunkVoxelData& subChunk = column.subChunks[subChunkIndex];
        if (!ShouldStoreSubChunk(subChunk)) {
            continue;
        }

        AppendBinary(payload, static_cast<std::uint8_t>(subChunkIndex));
        if (subChunk.isUniform) {
            AppendBinary(payload, kStoredSubChunkUniform);
            AppendBinary(payload, subChunk.uniformBlock);
        } else {
            AppendBinary(payload, kStoredSubChunkDense);
            const std::size_t byteCount = subChunk.blocks.size() * sizeof(std::uint16_t);
            const std::size_t offset = payload.size();
            payload.resize(offset + byteCount);
            std::memcpy(payload.data() + offset, subChunk.blocks.data(), byteCount);
        }
    }

    return payload;
}

void VoxelWorld::DeserializeChunkColumnPayload(std::istream& stream, ChunkColumnData& column) const {
    InitializeVoxelSubChunks(column);
    InitializeSubChunkMeshes(column, true);
    column.modified = false;

    std::uint32_t storedSubChunkCount = 0;
    ReadBinary(stream, storedSubChunkCount);

    for (std::uint32_t storedIndex = 0; storedIndex < storedSubChunkCount; ++storedIndex) {
        std::uint8_t subChunkIndex = 0;
        std::uint8_t storageMode = 0;
        ReadBinary(stream, subChunkIndex);
        ReadBinary(stream, storageMode);

        if (subChunkIndex >= static_cast<std::uint8_t>(kSubChunkCountY)) {
            throw std::runtime_error("Region subchunk index is out of range.");
        }

        SubChunkVoxelData& subChunk = column.subChunks[static_cast<std::size_t>(subChunkIndex)];
        if (storageMode == kStoredSubChunkUniform) {
            std::uint16_t blockValue = 0;
            ReadBinary(stream, blockValue);
            subChunk.blocks.clear();
            subChunk.uniformBlock = blockValue;
            subChunk.isUniform = true;
        } else if (storageMode == kStoredSubChunkDense) {
            subChunk.blocks.resize(static_cast<std::size_t>(kSubChunkVoxelCount));
            stream.read(
                reinterpret_cast<char*>(subChunk.blocks.data()),
                static_cast<std::streamsize>(subChunk.blocks.size() * sizeof(std::uint16_t))
            );
            subChunk.uniformBlock = 0;
            subChunk.isUniform = false;
        } else {
            throw std::runtime_error("Region subchunk storage mode is invalid.");
        }

        if (!stream) {
            throw std::runtime_error("Region chunk payload ended unexpectedly.");
        }
    }
}

void VoxelWorld::SaveRegionOverrides(
    int regionX,
    int regionZ,
    const std::unordered_map<std::int64_t, const ChunkColumnData*>& overrides
) const {
    struct SerializedRegionChunk {
        std::int32_t chunkX = 0;
        std::int32_t chunkZ = 0;
        std::vector<std::uint8_t> payload;
    };

    std::filesystem::create_directories(GetRegionDirectoryPath());

    std::vector<SerializedRegionChunk> serializedChunks;
    serializedChunks.reserve(overrides.size());
    std::unordered_set<std::int64_t> includedKeys;
    includedKeys.reserve(overrides.size());

    for (const auto& [key, column] : overrides) {
        if (column == nullptr) {
            continue;
        }

        const std::uint64_t unsignedKey = static_cast<std::uint64_t>(key);
        SerializedRegionChunk serializedChunk{};
        serializedChunk.chunkX = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey >> 32));
        serializedChunk.chunkZ = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey));
        serializedChunk.payload = SerializeChunkColumnPayload(*column);
        serializedChunks.push_back(std::move(serializedChunk));
        includedKeys.insert(key);
    }

    const std::filesystem::path regionPath = GetRegionFilePath(regionX, regionZ);
    if (std::filesystem::exists(regionPath)) {
        std::ifstream existingStream(regionPath, std::ios::binary);
        if (!existingStream.is_open()) {
            throw std::runtime_error("Failed to open region file.");
        }

        char magic[4] = {};
        std::int32_t fileRegionX = 0;
        std::int32_t fileRegionZ = 0;
        std::uint32_t chunkCount = 0;
        existingStream.read(magic, sizeof(magic));
        ReadBinary(existingStream, fileRegionX);
        ReadBinary(existingStream, fileRegionZ);
        ReadBinary(existingStream, chunkCount);

        if (!existingStream || std::memcmp(magic, kRegionMagic, sizeof(kRegionMagic)) != 0 ||
            fileRegionX != regionX || fileRegionZ != regionZ) {
            throw std::runtime_error("Region file is corrupted or incompatible.");
        }

        std::vector<RegionChunkIndexEntry> indexEntries(chunkCount);
        for (RegionChunkIndexEntry& indexEntry : indexEntries) {
            ReadBinary(existingStream, indexEntry.chunkX);
            ReadBinary(existingStream, indexEntry.chunkZ);
            ReadBinary(existingStream, indexEntry.offset);
            ReadBinary(existingStream, indexEntry.size);
            if (!existingStream) {
                throw std::runtime_error("Region index is corrupted.");
            }
        }

        for (const RegionChunkIndexEntry& indexEntry : indexEntries) {
            const std::int64_t key = MakeChunkKey(indexEntry.chunkX, indexEntry.chunkZ);
            if (includedKeys.contains(key)) {
                continue;
            }

            existingStream.seekg(static_cast<std::streamoff>(indexEntry.offset), std::ios::beg);
            if (!existingStream) {
                throw std::runtime_error("Region chunk offset is invalid.");
            }

            SerializedRegionChunk serializedChunk{};
            serializedChunk.chunkX = indexEntry.chunkX;
            serializedChunk.chunkZ = indexEntry.chunkZ;
            serializedChunk.payload.resize(indexEntry.size);
            existingStream.read(
                reinterpret_cast<char*>(serializedChunk.payload.data()),
                static_cast<std::streamsize>(serializedChunk.payload.size())
            );
            if (!existingStream) {
                throw std::runtime_error("Failed to read existing region chunk payload.");
            }

            serializedChunks.push_back(std::move(serializedChunk));
        }
    }

    std::sort(serializedChunks.begin(), serializedChunks.end(), [](const SerializedRegionChunk& lhs, const SerializedRegionChunk& rhs) {
        if (lhs.chunkZ != rhs.chunkZ) {
            return lhs.chunkZ < rhs.chunkZ;
        }
        return lhs.chunkX < rhs.chunkX;
    });

    std::ofstream stream(regionPath, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to write region file.");
    }

    stream.write(kRegionMagic, sizeof(kRegionMagic));
    WriteBinary(stream, static_cast<std::int32_t>(regionX));
    WriteBinary(stream, static_cast<std::int32_t>(regionZ));
    WriteBinary(stream, static_cast<std::uint32_t>(serializedChunks.size()));

    const std::uint64_t headerSize =
        sizeof(kRegionMagic) +
        sizeof(std::int32_t) +
        sizeof(std::int32_t) +
        sizeof(std::uint32_t);
    const std::uint64_t indexEntrySize =
        sizeof(std::int32_t) +
        sizeof(std::int32_t) +
        sizeof(std::uint64_t) +
        sizeof(std::uint32_t);
    std::uint64_t currentOffset = headerSize + indexEntrySize * static_cast<std::uint64_t>(serializedChunks.size());

    for (const SerializedRegionChunk& serializedChunk : serializedChunks) {
        WriteBinary(stream, serializedChunk.chunkX);
        WriteBinary(stream, serializedChunk.chunkZ);
        WriteBinary(stream, currentOffset);
        WriteBinary(stream, static_cast<std::uint32_t>(serializedChunk.payload.size()));
        currentOffset += static_cast<std::uint64_t>(serializedChunk.payload.size());
    }

    for (const SerializedRegionChunk& serializedChunk : serializedChunks) {
        stream.write(
            reinterpret_cast<const char*>(serializedChunk.payload.data()),
            static_cast<std::streamsize>(serializedChunk.payload.size())
        );
    }

    if (!stream) {
        throw std::runtime_error("Failed while saving region data.");
    }
}

void VoxelWorld::LoadRegionFile(int regionX, int regionZ, std::unordered_map<std::int64_t, ChunkColumnData>& outColumns) const {
    outColumns.clear();

    const std::filesystem::path regionPath = GetRegionFilePath(regionX, regionZ);
    if (!std::filesystem::exists(regionPath)) {
        return;
    }

    std::ifstream stream(regionPath, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open region file.");
    }

    char magic[4] = {};
    std::int32_t fileRegionX = 0;
    std::int32_t fileRegionZ = 0;
    std::uint32_t chunkCount = 0;
    stream.read(magic, sizeof(magic));
    ReadBinary(stream, fileRegionX);
    ReadBinary(stream, fileRegionZ);
    ReadBinary(stream, chunkCount);

    if (!stream || std::memcmp(magic, kRegionMagic, sizeof(kRegionMagic)) != 0 ||
        fileRegionX != regionX || fileRegionZ != regionZ) {
        throw std::runtime_error("Region file is corrupted or incompatible.");
    }

    std::vector<RegionChunkIndexEntry> indexEntries(chunkCount);
    for (RegionChunkIndexEntry& indexEntry : indexEntries) {
        ReadBinary(stream, indexEntry.chunkX);
        ReadBinary(stream, indexEntry.chunkZ);
        ReadBinary(stream, indexEntry.offset);
        ReadBinary(stream, indexEntry.size);
        if (!stream) {
            throw std::runtime_error("Region index is corrupted.");
        }
    }

    for (const RegionChunkIndexEntry& indexEntry : indexEntries) {
        stream.seekg(static_cast<std::streamoff>(indexEntry.offset), std::ios::beg);
        if (!stream) {
            throw std::runtime_error("Region chunk offset is invalid.");
        }

        ChunkColumnData column{};
        DeserializeChunkColumnPayload(stream, column);
        outColumns.emplace(MakeChunkKey(indexEntry.chunkX, indexEntry.chunkZ), std::move(column));
    }
}

void VoxelWorld::SaveRegionFile(int regionX, int regionZ, const std::unordered_map<std::int64_t, ChunkColumnData>& columns) const {
    std::filesystem::create_directories(GetRegionDirectoryPath());

    std::ofstream stream(GetRegionFilePath(regionX, regionZ), std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to write region file.");
    }

    stream.write(kRegionMagic, sizeof(kRegionMagic));
    WriteBinary(stream, static_cast<std::int32_t>(regionX));
    WriteBinary(stream, static_cast<std::int32_t>(regionZ));
    WriteBinary(stream, static_cast<std::uint32_t>(columns.size()));

    struct SerializedRegionChunk {
        std::int32_t chunkX = 0;
        std::int32_t chunkZ = 0;
        std::vector<std::uint8_t> payload;
    };

    std::vector<SerializedRegionChunk> serializedChunks;
    serializedChunks.reserve(columns.size());
    for (const auto& [key, column] : columns) {
        const std::uint64_t unsignedKey = static_cast<std::uint64_t>(key);
        SerializedRegionChunk serializedChunk{};
        serializedChunk.chunkX = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey >> 32));
        serializedChunk.chunkZ = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey));
        serializedChunk.payload = SerializeChunkColumnPayload(column);
        serializedChunks.push_back(std::move(serializedChunk));
    }

    std::sort(serializedChunks.begin(), serializedChunks.end(), [](const SerializedRegionChunk& lhs, const SerializedRegionChunk& rhs) {
        if (lhs.chunkZ != rhs.chunkZ) {
            return lhs.chunkZ < rhs.chunkZ;
        }
        return lhs.chunkX < rhs.chunkX;
    });

    const std::uint64_t headerSize =
        sizeof(kRegionMagic) +
        sizeof(std::int32_t) +
        sizeof(std::int32_t) +
        sizeof(std::uint32_t);
    const std::uint64_t indexEntrySize =
        sizeof(std::int32_t) +
        sizeof(std::int32_t) +
        sizeof(std::uint64_t) +
        sizeof(std::uint32_t);
    std::uint64_t currentOffset = headerSize + indexEntrySize * static_cast<std::uint64_t>(serializedChunks.size());

    for (const SerializedRegionChunk& serializedChunk : serializedChunks) {
        WriteBinary(stream, serializedChunk.chunkX);
        WriteBinary(stream, serializedChunk.chunkZ);
        WriteBinary(stream, currentOffset);
        WriteBinary(stream, static_cast<std::uint32_t>(serializedChunk.payload.size()));
        currentOffset += static_cast<std::uint64_t>(serializedChunk.payload.size());
    }

    for (const SerializedRegionChunk& serializedChunk : serializedChunks) {
        stream.write(
            reinterpret_cast<const char*>(serializedChunk.payload.data()),
            static_cast<std::streamsize>(serializedChunk.payload.size())
        );
    }

    if (!stream) {
        throw std::runtime_error("Failed while saving region data.");
    }
}

std::uint16_t VoxelWorld::GetBlock(int worldX, int worldY, int worldZ) {
    if (worldY < 0 || worldY >= kWorldSizeY) {
        return 0;
    }

    EnsureInitialized();

    const int chunkX = FloorDiv(worldX, kChunkSizeX);
    const int chunkZ = FloorDiv(worldZ, kChunkSizeZ);
    const int localX = PositiveMod(worldX, kChunkSizeX);
    const int localZ = PositiveMod(worldZ, kChunkSizeZ);
    const int subChunkIndex = GetSubChunkIndex(worldY);
    const int localY = worldY - subChunkIndex * kSubChunkSize;

    const auto columnIt = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (columnIt == chunkColumns_.end() || columnIt->second.subChunks.size() != kSubChunkCountY) {
        return SampleGeneratedBlock(worldY);
    }

    return GetSubChunkBlock(
        columnIt->second.subChunks[static_cast<std::size_t>(subChunkIndex)],
        localX,
        localY,
        localZ
    );
}

bool VoxelWorld::SetBlock(int worldX, int worldY, int worldZ, std::uint16_t blockValue) {
    if (worldY < 0 || worldY >= kWorldSizeY) {
        return false;
    }

    EnsureInitialized();

    const int chunkX = FloorDiv(worldX, kChunkSizeX);
    const int chunkZ = FloorDiv(worldZ, kChunkSizeZ);
    const int localX = PositiveMod(worldX, kChunkSizeX);
    const int localZ = PositiveMod(worldZ, kChunkSizeZ);
    const int subChunkIndex = GetSubChunkIndex(worldY);
    const int localY = worldY - subChunkIndex * kSubChunkSize;

    EnsureChunkColumn(chunkX, chunkZ);
    auto columnIt = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (columnIt == chunkColumns_.end()) {
        return false;
    }

    if (columnIt->second.subChunks.size() != kSubChunkCountY) {
        InitializeVoxelSubChunks(columnIt->second);
    }

    SubChunkVoxelData& subChunk = columnIt->second.subChunks[static_cast<std::size_t>(subChunkIndex)];
    if (GetSubChunkBlock(subChunk, localX, localY, localZ) == blockValue) {
        return false;
    }

    SetSubChunkBlock(subChunk, localX, localY, localZ, blockValue);
    TryCollapseSubChunk(subChunk);

    columnIt->second.modified = true;
    saveDirty_ = true;

    MarkSubChunkDirty(chunkX, chunkZ, subChunkIndex);
    if (worldY > 0 && worldY % kSubChunkSize == 0) {
        MarkSubChunkDirty(chunkX, chunkZ, subChunkIndex - 1);
    }
    if (worldY + 1 < kWorldSizeY && (worldY + 1) % kSubChunkSize == 0) {
        MarkSubChunkDirty(chunkX, chunkZ, subChunkIndex + 1);
    }

    if (localX == 0) {
        EnsureChunkColumn(chunkX - 1, chunkZ);
        MarkSubChunkDirty(chunkX - 1, chunkZ, subChunkIndex);
    }
    if (localX == kChunkSizeX - 1) {
        EnsureChunkColumn(chunkX + 1, chunkZ);
        MarkSubChunkDirty(chunkX + 1, chunkZ, subChunkIndex);
    }
    if (localZ == 0) {
        EnsureChunkColumn(chunkX, chunkZ - 1);
        MarkSubChunkDirty(chunkX, chunkZ - 1, subChunkIndex);
    }
    if (localZ == kChunkSizeZ - 1) {
        EnsureChunkColumn(chunkX, chunkZ + 1);
        MarkSubChunkDirty(chunkX, chunkZ + 1, subChunkIndex);
    }

    return true;
}

bool VoxelWorld::CopyChunkMeshBatch(int chunkX, int chunkZ, ChunkMeshBatchData& outBatch) const {
    const auto columnIt = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (columnIt == chunkColumns_.end() || columnIt->second.subChunkMeshes.size() != kSubChunkCountY) {
        return false;
    }

    outBatch.id = {chunkX, chunkZ};
    outBatch.vertices.clear();
    outBatch.indices.clear();

    std::size_t totalVertexCount = 0;
    std::size_t totalIndexCount = 0;
    for (const SubChunkMeshData& subChunkMesh : columnIt->second.subChunkMeshes) {
        totalVertexCount += subChunkMesh.vertices.size();
        totalIndexCount += subChunkMesh.indices.size();
    }

    if (totalVertexCount == 0 || totalIndexCount == 0) {
        return false;
    }

    outBatch.vertices.reserve(totalVertexCount);
    outBatch.indices.reserve(totalIndexCount);

    for (const SubChunkMeshData& subChunkMesh : columnIt->second.subChunkMeshes) {
        if (subChunkMesh.vertices.empty() || subChunkMesh.indices.empty()) {
            continue;
        }

        const std::uint32_t vertexOffset = static_cast<std::uint32_t>(outBatch.vertices.size());
        outBatch.vertices.insert(outBatch.vertices.end(), subChunkMesh.vertices.begin(), subChunkMesh.vertices.end());
        for (std::uint32_t index : subChunkMesh.indices) {
            outBatch.indices.push_back(vertexOffset + index);
        }
    }

    return !outBatch.vertices.empty() && !outBatch.indices.empty();
}

void VoxelWorld::Save() {
    if (!initialized_ || !saveDirty_) {
        return;
    }

    SaveAllDirtyChunks();
}

bool VoxelWorld::IsChunkInsideFrustum(
    int chunkX,
    int chunkZ,
    const Vec3& cameraPosition,
    const Vec3& cameraForward,
    float verticalFovDegrees,
    float aspectRatio
) {
    Vec3 forward = Normalize(cameraForward);
    if (Length(forward) <= 0.00001f) {
        forward = {0.0f, 0.0f, -1.0f};
    }

    const float paddedVerticalFov = verticalFovDegrees + kFrustumPaddingDegrees;
    const Mat4 projection = Perspective(paddedVerticalFov * 3.14159265358979323846f / 180.0f, aspectRatio, 0.1f, 2048.0f);
    const Mat4 view = LookAt(cameraPosition, cameraPosition + forward, {0.0f, 1.0f, 0.0f});
    const Mat4 viewProjection = Multiply(projection, view);

    const float minX = static_cast<float>(chunkX * kChunkSizeX);
    const float maxX = minX + static_cast<float>(kChunkSizeX);
    const float minY = 0.0f;
    const float maxY = static_cast<float>(kWorldSizeY);
    const float minZ = static_cast<float>(chunkZ * kChunkSizeZ);
    const float maxZ = minZ + static_cast<float>(kChunkSizeZ);

    const std::array<Vec3, 8> corners = {{
        {minX, minY, minZ},
        {maxX, minY, minZ},
        {minX, maxY, minZ},
        {maxX, maxY, minZ},
        {minX, minY, maxZ},
        {maxX, minY, maxZ},
        {minX, maxY, maxZ},
        {maxX, maxY, maxZ},
    }};

    std::array<ClipVertex, 8> clipCorners{};
    for (std::size_t i = 0; i < corners.size(); ++i) {
        clipCorners[i] = TransformPoint(viewProjection, corners[i]);
    }

    auto allOutside = [&clipCorners](auto predicate) {
        for (const ClipVertex& corner : clipCorners) {
            if (!predicate(corner)) {
                return false;
            }
        }
        return true;
    };

    if (allOutside([](const ClipVertex& v) { return v.x < -v.w; })) {
        return false;
    }
    if (allOutside([](const ClipVertex& v) { return v.x > v.w; })) {
        return false;
    }
    if (allOutside([](const ClipVertex& v) { return v.y < -v.w; })) {
        return false;
    }
    if (allOutside([](const ClipVertex& v) { return v.y > v.w; })) {
        return false;
    }
    if (allOutside([](const ClipVertex& v) { return v.z < 0.0f; })) {
        return false;
    }
    if (allOutside([](const ClipVertex& v) { return v.z > v.w; })) {
        return false;
    }

    return true;
}

void VoxelWorld::BuildVisibleMesh(
    int centerChunkX,
    int centerChunkZ,
    int renderRadius,
    const Vec3& cameraPosition,
    const Vec3& cameraForward,
    float verticalFovDegrees,
    float aspectRatio,
    WorldMeshData& outMesh
) {
    (void)cameraPosition;
    (void)cameraForward;
    (void)verticalFovDegrees;
    (void)aspectRatio;

    EnsureInitialized();

    outMesh.batches.clear();
    outMesh.totalVertexCount = 0;
    outMesh.totalIndexCount = 0;

    for (int dz = -renderRadius; dz <= renderRadius; ++dz) {
        for (int dx = -renderRadius; dx <= renderRadius; ++dx) {
            if (dx * dx + dz * dz > renderRadius * renderRadius) {
                continue;
            }

            const int chunkX = centerChunkX + dx;
            const int chunkZ = centerChunkZ + dz;

            auto columnIt = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
            if (columnIt == chunkColumns_.end()) {
                continue;
            }

            if (columnIt->second.subChunkMeshes.size() != kSubChunkCountY) {
                continue;
            }

            for (int subChunkIndex = 0; subChunkIndex < kSubChunkCountY; ++subChunkIndex) {
                const SubChunkMeshData& subChunkMesh = columnIt->second.subChunkMeshes[static_cast<std::size_t>(subChunkIndex)];
                if (subChunkMesh.vertices.empty() || subChunkMesh.indices.empty()) {
                    continue;
                }

                WorldVisibleBatch batch{};
                batch.id = {chunkX, chunkZ, subChunkIndex};
                batch.revision = subChunkMesh.revision;

                outMesh.totalVertexCount += static_cast<std::uint32_t>(subChunkMesh.vertices.size());
                outMesh.totalIndexCount += static_cast<std::uint32_t>(subChunkMesh.indices.size());
                outMesh.batches.push_back(std::move(batch));
            }
        }
    }

    outMesh.loadedChunkCount = chunkColumns_.size();
}

std::size_t VoxelWorld::GetLoadedChunkCount() const {
    return chunkColumns_.size();
}

const TerrainConfig& VoxelWorld::GetTerrainConfig() const {
    return terrainConfig_;
}

void VoxelWorld::EnsureTerrainConfigLoaded() {
    if (terrainConfigLoaded_) {
        return;
    }

    terrainConfig_ = LoadTerrainConfig(std::string(ASSET_DIR) + "/assets/config/terrain.json");
    terrainConfigLoaded_ = true;
}

void VoxelWorld::AppendFaceQuad(
    SubChunkMeshData& mesh,
    float worldX,
    float worldY,
    float worldZ,
    int faceIndex
) const {
    const float x = worldX;
    const float y = worldY;
    const float z = worldZ;

    switch (faceIndex) {
    case 0:
        AppendQuad(mesh, {x, y + 1.0f, z}, {x, y + 1.0f, z + 1.0f}, {x + 1.0f, y + 1.0f, z + 1.0f}, {x + 1.0f, y + 1.0f, z}, 1.0f, 1.0f);
        break;
    case 1:
        AppendQuad(mesh, {x, y, z + 1.0f}, {x, y, z}, {x + 1.0f, y, z}, {x + 1.0f, y, z + 1.0f}, 1.0f, 1.0f);
        break;
    case 2:
        AppendQuad(mesh, {x + 1.0f, y, z}, {x, y, z}, {x, y + 1.0f, z}, {x + 1.0f, y + 1.0f, z}, 1.0f, 1.0f);
        break;
    case 3:
        AppendQuad(mesh, {x, y, z + 1.0f}, {x + 1.0f, y, z + 1.0f}, {x + 1.0f, y + 1.0f, z + 1.0f}, {x, y + 1.0f, z + 1.0f}, 1.0f, 1.0f);
        break;
    case 4:
        AppendQuad(mesh, {x, y, z}, {x, y, z + 1.0f}, {x, y + 1.0f, z + 1.0f}, {x, y + 1.0f, z}, 1.0f, 1.0f);
        break;
    case 5:
        AppendQuad(mesh, {x + 1.0f, y, z + 1.0f}, {x + 1.0f, y, z}, {x + 1.0f, y + 1.0f, z}, {x + 1.0f, y + 1.0f, z + 1.0f}, 1.0f, 1.0f);
        break;
    default:
        break;
    }
}

void VoxelWorld::RebuildSubChunkMesh(int chunkX, int chunkZ, int subChunkIndex) {
    auto columnIt = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (columnIt == chunkColumns_.end()) {
        return;
    }

    ChunkColumnData& column = columnIt->second;
    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        InitializeSubChunkMeshes(column, true);
    }
    if (column.subChunks.size() != kSubChunkCountY) {
        InitializeVoxelSubChunks(column);
    }

    SubChunkMeshData& mesh = column.subChunkMeshes[static_cast<std::size_t>(subChunkIndex)];
    const SubChunkVoxelData& voxelSubChunk = column.subChunks[static_cast<std::size_t>(subChunkIndex)];
    mesh.vertices.clear();
    mesh.indices.clear();

    const int minWorldY = subChunkIndex * kSubChunkSize;
    const int maxWorldY = std::min(minWorldY + kSubChunkSize, kWorldSizeY);
    const int minWorldX = chunkX * kChunkSizeX;
    const int minWorldZ = chunkZ * kChunkSizeZ;

    if (voxelSubChunk.isUniform && voxelSubChunk.uniformBlock == 0) {
        mesh.dirty = false;
        ++mesh.revision;
        return;
    }

    constexpr int dims[3] = {kChunkSizeX, kSubChunkSize, kChunkSizeZ};
    std::vector<int> mask(static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ));

    const auto sampleBlock = [&](int localX, int localY, int localZ) -> std::uint16_t {
        if (localX >= 0 && localX < kChunkSizeX &&
            localY >= 0 && localY < kSubChunkSize &&
            localZ >= 0 && localZ < kChunkSizeZ) {
            return GetSubChunkBlock(voxelSubChunk, localX, localY, localZ);
        }

        return GetBlock(minWorldX + localX, minWorldY + localY, minWorldZ + localZ);
    };

    const auto emitGreedyQuad = [&](int axis, bool positiveNormal, int slice, int startU, int startV, int width, int height) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;

        Vec3 base{};
        if (axis == 0) {
            base.x = static_cast<float>(minWorldX + slice);
        } else if (axis == 1) {
            base.y = static_cast<float>(minWorldY + slice);
        } else {
            base.z = static_cast<float>(minWorldZ + slice);
        }

        if (u == 0) {
            base.x = static_cast<float>(minWorldX + startU);
        } else if (u == 1) {
            base.y = static_cast<float>(minWorldY + startU);
        } else {
            base.z = static_cast<float>(minWorldZ + startU);
        }

        if (v == 0) {
            base.x = static_cast<float>(minWorldX + startV);
        } else if (v == 1) {
            base.y = static_cast<float>(minWorldY + startV);
        } else {
            base.z = static_cast<float>(minWorldZ + startV);
        }

        Vec3 du{};
        Vec3 dv{};
        if (u == 0) {
            du.x = static_cast<float>(width);
        } else if (u == 1) {
            du.y = static_cast<float>(width);
        } else {
            du.z = static_cast<float>(width);
        }

        if (v == 0) {
            dv.x = static_cast<float>(height);
        } else if (v == 1) {
            dv.y = static_cast<float>(height);
        } else {
            dv.z = static_cast<float>(height);
        }

        const Vec3 p0 = base;
        const Vec3 p1 = positiveNormal ? base + du : base + dv;
        const Vec3 p2 = base + du + dv;
        const Vec3 p3 = positiveNormal ? base + dv : base + du;
        AppendQuad(mesh, p0, p1, p2, p3, static_cast<float>(width), static_cast<float>(height));
    };

    for (int axis = 0; axis < 3; ++axis) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;

        for (int slice = -1; slice < dims[axis]; ++slice) {
            int maskIndex = 0;
            for (int j = 0; j < dims[v]; ++j) {
                for (int i = 0; i < dims[u]; ++i) {
                    int aCoords[3] = {};
                    int bCoords[3] = {};
                    aCoords[axis] = slice;
                    bCoords[axis] = slice + 1;
                    aCoords[u] = i;
                    bCoords[u] = i;
                    aCoords[v] = j;
                    bCoords[v] = j;

                    const std::uint16_t a = sampleBlock(aCoords[0], aCoords[1], aCoords[2]);
                    const std::uint16_t b = sampleBlock(bCoords[0], bCoords[1], bCoords[2]);

                    if (a != 0 && b == 0) {
                        mask[static_cast<std::size_t>(maskIndex)] = static_cast<int>(a);
                    } else if (a == 0 && b != 0) {
                        mask[static_cast<std::size_t>(maskIndex)] = -static_cast<int>(b);
                    } else {
                        mask[static_cast<std::size_t>(maskIndex)] = 0;
                    }

                    ++maskIndex;
                }
            }

            for (int j = 0; j < dims[v]; ++j) {
                for (int i = 0; i < dims[u];) {
                    const int current = mask[static_cast<std::size_t>(i + j * dims[u])];
                    if (current == 0) {
                        ++i;
                        continue;
                    }

                    int width = 1;
                    while (i + width < dims[u] &&
                           mask[static_cast<std::size_t>(i + width + j * dims[u])] == current) {
                        ++width;
                    }

                    int height = 1;
                    bool done = false;
                    while (j + height < dims[v] && !done) {
                        for (int k = 0; k < width; ++k) {
                            if (mask[static_cast<std::size_t>(i + k + (j + height) * dims[u])] != current) {
                                done = true;
                                break;
                            }
                        }
                        if (!done) {
                            ++height;
                        }
                    }

                    emitGreedyQuad(axis, current > 0, slice + 1, i, j, width, height);

                    for (int row = 0; row < height; ++row) {
                        for (int col = 0; col < width; ++col) {
                            mask[static_cast<std::size_t>(i + col + (j + row) * dims[u])] = 0;
                        }
                    }

                    i += width;
                }
            }
        }
    }

    mesh.dirty = false;
    ++mesh.revision;
}
