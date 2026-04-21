#include "BlockRegistry.h"
#include "VoxelWorld.h"
#include "WorldGenerator.h"

#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
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

using Clock = std::chrono::steady_clock;

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
constexpr std::uint32_t kFaceRowBitMask = (1u << kSubChunkSize) - 1u;

struct RuntimeProfileAccumulator {
    std::atomic<std::uint64_t> totalNs{0};
    std::atomic<std::uint64_t> maxNs{0};
    std::atomic<std::uint32_t> samples{0};
};

struct RuntimeCountAccumulator {
    std::atomic<std::uint64_t> totalValue{0};
    std::atomic<std::uint64_t> maxValue{0};
    std::atomic<std::uint32_t> samples{0};
};

RuntimeProfileAccumulator gChunkLoadProfile{};
RuntimeProfileAccumulator gDiskLoadProfile{};
RuntimeProfileAccumulator gGenerateProfile{};
RuntimeProfileAccumulator gMeshBuildProfile{};
RuntimeProfileAccumulator gSaveProfile{};
RuntimeProfileAccumulator gUnloadProfile{};
RuntimeProfileAccumulator gSaveFileProfile{};
RuntimeProfileAccumulator gGetBlockProfile{};
RuntimeProfileAccumulator gGeneratedBlockProfile{};
RuntimeCountAccumulator gUnloadCountProfile{};
RuntimeCountAccumulator gSaveCountProfile{};

void RecordRuntimeProfileSample(RuntimeProfileAccumulator& accumulator, std::uint64_t elapsedNs) {
    accumulator.totalNs.fetch_add(elapsedNs, std::memory_order_relaxed);
    accumulator.samples.fetch_add(1u, std::memory_order_relaxed);
    std::uint64_t observedMax = accumulator.maxNs.load(std::memory_order_relaxed);
    while (observedMax < elapsedNs &&
           !accumulator.maxNs.compare_exchange_weak(
               observedMax,
               elapsedNs,
               std::memory_order_relaxed,
               std::memory_order_relaxed
           )) {
    }
}

RuntimeProfileStage ConsumeRuntimeProfileStage(RuntimeProfileAccumulator& accumulator) {
    RuntimeProfileStage stage{};
    const std::uint64_t totalNs = accumulator.totalNs.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t maxNs = accumulator.maxNs.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t samples = accumulator.samples.exchange(0u, std::memory_order_relaxed);
    stage.samples = samples;
    if (samples > 0) {
        stage.averageMs = static_cast<double>(totalNs) / static_cast<double>(samples) / 1'000'000.0;
        stage.maxMs = static_cast<double>(maxNs) / 1'000'000.0;
    }
    return stage;
}

void RecordRuntimeCountSample(RuntimeCountAccumulator& accumulator, std::uint64_t value) {
    accumulator.totalValue.fetch_add(value, std::memory_order_relaxed);
    accumulator.samples.fetch_add(1u, std::memory_order_relaxed);
    std::uint64_t observedMax = accumulator.maxValue.load(std::memory_order_relaxed);
    while (observedMax < value &&
           !accumulator.maxValue.compare_exchange_weak(
               observedMax,
               value,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

RuntimeProfileStage ConsumeRuntimeCountStage(RuntimeCountAccumulator& accumulator) {
    RuntimeProfileStage stage{};
    const std::uint64_t totalValue = accumulator.totalValue.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t maxValue = accumulator.maxValue.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t samples = accumulator.samples.exchange(0u, std::memory_order_relaxed);
    stage.samples = samples;
    if (samples > 0) {
        stage.averageMs = static_cast<double>(totalValue) / static_cast<double>(samples);
        stage.maxMs = static_cast<double>(maxValue);
    }
    return stage;
}

struct RegionChunkIndexEntry {
    std::int32_t chunkX = 0;
    std::int32_t chunkZ = 0;
    std::uint64_t offset = 0;
    std::uint32_t size = 0;
};

struct FaceMaskCell {
    std::uint32_t packed = 0;

    bool operator==(const FaceMaskCell& other) const = default;
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

Vec3 SubtractVec3(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

int CountTrailingZeros(std::uint32_t value) {
    return static_cast<int>(std::countr_zero(value));
}

std::uint8_t PackAoPattern(const std::array<std::uint8_t, 4>& aoPattern) {
    return static_cast<std::uint8_t>(
        (aoPattern[0] & 0x3u) |
        ((aoPattern[1] & 0x3u) << 2u) |
        ((aoPattern[2] & 0x3u) << 4u) |
        ((aoPattern[3] & 0x3u) << 6u)
    );
}

FaceMaskCell MakeFaceMaskCell(std::uint16_t materialId, bool positiveNormal, std::uint8_t packedAo) {
    FaceMaskCell cell{};
    cell.packed =
        (static_cast<std::uint32_t>(materialId) & 0xFFFFu) |
        ((static_cast<std::uint32_t>(packedAo) & 0xFFu) << 16u) |
        ((positiveNormal ? 1u : 0u) << 24u);
    return cell;
}

bool IsPositiveFaceMaskCell(const FaceMaskCell& cell) {
    return (cell.packed & (1u << 24u)) != 0;
}

std::uint16_t GetFaceMaskCellMaterialId(const FaceMaskCell& cell) {
    return static_cast<std::uint16_t>(cell.packed & 0xFFFFu);
}

Vec3 CrossVec3(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float DotVec3(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

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
    float vMax,
    const std::array<float, 4>& ao = {1.0f, 1.0f, 1.0f, 1.0f},
    bool flipDiagonal = false,
    bool reverseWinding = false
) {
    const std::uint32_t baseIndex = static_cast<std::uint32_t>(mesh.vertices.size());

    mesh.vertices.push_back({{p0.x, p0.y, p0.z}, {0.0f, 0.0f}, ao[0]});
    mesh.vertices.push_back({{p1.x, p1.y, p1.z}, {uMax, 0.0f}, ao[1]});
    mesh.vertices.push_back({{p2.x, p2.y, p2.z}, {uMax, vMax}, ao[2]});
    mesh.vertices.push_back({{p3.x, p3.y, p3.z}, {0.0f, vMax}, ao[3]});

    if (flipDiagonal) {
        if (reverseWinding) {
            mesh.indices.push_back(baseIndex + 0);
            mesh.indices.push_back(baseIndex + 3);
            mesh.indices.push_back(baseIndex + 1);
            mesh.indices.push_back(baseIndex + 1);
            mesh.indices.push_back(baseIndex + 3);
            mesh.indices.push_back(baseIndex + 2);
        } else {
            mesh.indices.push_back(baseIndex + 0);
            mesh.indices.push_back(baseIndex + 1);
            mesh.indices.push_back(baseIndex + 3);
            mesh.indices.push_back(baseIndex + 1);
            mesh.indices.push_back(baseIndex + 2);
            mesh.indices.push_back(baseIndex + 3);
        }
    } else {
        if (reverseWinding) {
            mesh.indices.push_back(baseIndex + 0);
            mesh.indices.push_back(baseIndex + 2);
            mesh.indices.push_back(baseIndex + 1);
            mesh.indices.push_back(baseIndex + 0);
            mesh.indices.push_back(baseIndex + 3);
            mesh.indices.push_back(baseIndex + 2);
        } else {
            mesh.indices.push_back(baseIndex + 0);
            mesh.indices.push_back(baseIndex + 1);
            mesh.indices.push_back(baseIndex + 2);
            mesh.indices.push_back(baseIndex + 0);
            mesh.indices.push_back(baseIndex + 2);
            mesh.indices.push_back(baseIndex + 3);
        }
    }
}

template <typename MeshType>
void AppendQuadRecord(
    MeshType& mesh,
    int localCellX,
    int localCellY,
    int localCellZ,
    int subChunkIndex,
    int axis,
    bool positiveNormal,
    int width,
    int height,
    std::uint16_t materialId,
    const std::array<std::uint8_t, 4>& aoPattern,
    bool flipDiagonal,
    bool reverseWinding
) {
    WorldQuadRecord record{};
    record.packed0 =
        (static_cast<std::uint32_t>(localCellX) & 0xFu) |
        ((static_cast<std::uint32_t>(localCellY) & 0xFu) << 4u) |
        ((static_cast<std::uint32_t>(localCellZ) & 0xFu) << 8u) |
        ((static_cast<std::uint32_t>(subChunkIndex) & 0x1Fu) << 12u) |
        ((static_cast<std::uint32_t>(axis) & 0x3u) << 17u) |
        ((positiveNormal ? 1u : 0u) << 19u) |
        ((static_cast<std::uint32_t>(width - 1) & 0xFu) << 20u) |
        ((static_cast<std::uint32_t>(height - 1) & 0xFu) << 24u);
    if (flipDiagonal) {
        record.packed0 |= (1u << 28u);
    }
    if (reverseWinding) {
        record.packed0 |= (1u << 29u);
    }

    record.packed1 =
        (static_cast<std::uint32_t>(aoPattern[0]) & 0x3u) |
        ((static_cast<std::uint32_t>(aoPattern[1]) & 0x3u) << 2u) |
        ((static_cast<std::uint32_t>(aoPattern[2]) & 0x3u) << 4u) |
        ((static_cast<std::uint32_t>(aoPattern[3]) & 0x3u) << 6u) |
        ((static_cast<std::uint32_t>(materialId) & 0xFFFFu) << 8u);

    mesh.push_back(record);
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

std::uint8_t ComputeVertexAoValue(bool side1, bool side2, bool corner) {
    if (side1 && side2) {
        return 0;
    }

    return static_cast<std::uint8_t>(3 - (static_cast<int>(side1) + static_cast<int>(side2) + static_cast<int>(corner)));
}

template <typename SampleFn>
std::array<std::uint8_t, 4> ComputeQuadAoPattern(
    const SampleFn& sampleBlock,
    int axis,
    bool positiveNormal,
    int planeSlice,
    int startU,
    int startV,
    int width,
    int height
) {
    const int u = (axis + 1) % 3;
    const int v = (axis + 2) % 3;
    const int airAxis = positiveNormal ? planeSlice : (planeSlice - 1);
    const int uMin = startU;
    const int uMax = startU + width - 1;
    const int vMin = startV;
    const int vMax = startV + height - 1;

    const auto sampleAo = [&](int faceU, int faceV, int du, int dv) {
        int coords[3] = {};
        coords[axis] = airAxis;
        coords[u] = faceU;
        coords[v] = faceV;

        const bool side1 = sampleBlock(
            coords[0] + (u == 0 ? du : 0),
            coords[1] + (u == 1 ? du : 0),
            coords[2] + (u == 2 ? du : 0)) != 0;
        const bool side2 = sampleBlock(
            coords[0] + (v == 0 ? dv : 0),
            coords[1] + (v == 1 ? dv : 0),
            coords[2] + (v == 2 ? dv : 0)) != 0;
        const bool corner = sampleBlock(
            coords[0] + (u == 0 ? du : 0) + (v == 0 ? dv : 0),
            coords[1] + (u == 1 ? du : 0) + (v == 1 ? dv : 0),
            coords[2] + (u == 2 ? du : 0) + (v == 2 ? dv : 0)) != 0;
        return ComputeVertexAoValue(side1, side2, corner);
    };

    return {
        sampleAo(uMin, vMin, -1, -1),
        sampleAo(uMax, vMin, 1, -1),
        sampleAo(uMax, vMax, 1, 1),
        sampleAo(uMin, vMax, -1, 1),
    };
}

template <typename SampleFn>
void BuildSubChunkQuadRecords(
    const SampleFn& sampleBlock,
    int minWorldX,
    int minWorldY,
    int minWorldZ,
    int subChunkIndex,
    std::vector<WorldQuadRecord>& outQuads
) {
    const auto profileStartTime = Clock::now();
    constexpr int dims[3] = {kChunkSizeX, kSubChunkSize, kChunkSizeZ};
    std::vector<FaceMaskCell> mask(static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ));

    const auto emitGreedyQuad = [&](
                                    int axis,
                                    bool positiveNormal,
                                    int slice,
                                    int startU,
                                    int startV,
                                    int width,
                                    int height,
                                    int materialValue) {
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

        Vec3 p0{};
        Vec3 p1{};
        Vec3 p2{};
        Vec3 p3{};
        const std::array<std::uint8_t, 4> baseAoPattern =
            ComputeQuadAoPattern(sampleBlock, axis, positiveNormal, slice, startU, startV, width, height);
        std::array<std::uint8_t, 4> vertexAoPattern{};
        const std::uint16_t materialId = static_cast<std::uint16_t>(std::abs(materialValue));

        if (axis == 0) {
            if (positiveNormal) {
                p0 = base + dv;
                p1 = base;
                p2 = base + du;
                p3 = base + du + dv;
                vertexAoPattern = {baseAoPattern[3], baseAoPattern[0], baseAoPattern[1], baseAoPattern[2]};
            } else {
                p0 = base;
                p1 = base + dv;
                p2 = base + du + dv;
                p3 = base + du;
                vertexAoPattern = {baseAoPattern[0], baseAoPattern[3], baseAoPattern[2], baseAoPattern[1]};
            }
        } else if (positiveNormal) {
            p0 = base;
            p1 = base + du;
            p2 = base + du + dv;
            p3 = base + dv;
            vertexAoPattern = baseAoPattern;
        } else {
            p0 = base + du;
            p1 = base;
            p2 = base + dv;
            p3 = base + du + dv;
            vertexAoPattern = {baseAoPattern[1], baseAoPattern[0], baseAoPattern[3], baseAoPattern[2]};
        }

        const bool flipDiagonal =
            (static_cast<int>(vertexAoPattern[0]) + static_cast<int>(vertexAoPattern[2])) >
            (static_cast<int>(vertexAoPattern[1]) + static_cast<int>(vertexAoPattern[3]));
        Vec3 expectedNormal{};
        if (axis == 0) {
            expectedNormal.x = positiveNormal ? 1.0f : -1.0f;
        } else if (axis == 1) {
            expectedNormal.y = positiveNormal ? 1.0f : -1.0f;
        } else {
            expectedNormal.z = positiveNormal ? 1.0f : -1.0f;
        }
        const Vec3 edge01 = SubtractVec3(p1, p0);
        const Vec3 edge02 = SubtractVec3(p2, p0);
        const bool reverseWinding = DotVec3(CrossVec3(edge01, edge02), expectedNormal) < 0.0f;
        int localCellCoords[3] = {};
        localCellCoords[axis] = positiveNormal ? (slice - 1) : slice;
        localCellCoords[u] = startU;
        localCellCoords[v] = startV;
        AppendQuadRecord(
            outQuads,
            localCellCoords[0],
            localCellCoords[1],
            localCellCoords[2],
            subChunkIndex,
            axis,
            positiveNormal,
            width,
            height,
            materialId,
            vertexAoPattern,
            flipDiagonal,
            reverseWinding
        );
    };

    for (int axis = 0; axis < 3; ++axis) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;

        for (int slice = -1; slice < dims[axis]; ++slice) {
            std::array<std::uint32_t, kSubChunkSize> rowBits{};
            for (int j = 0; j < dims[v]; ++j) {
                std::array<std::uint16_t, kSubChunkSize> rowValuesA{};
                std::array<std::uint16_t, kSubChunkSize> rowValuesB{};
                std::uint32_t solidMaskA = 0;
                std::uint32_t solidMaskB = 0;

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
                    rowValuesA[static_cast<std::size_t>(i)] = a;
                    rowValuesB[static_cast<std::size_t>(i)] = b;
                    if (a != 0) {
                        solidMaskA |= (1u << i);
                    }
                    if (b != 0) {
                        solidMaskB |= (1u << i);
                    }
                    mask[static_cast<std::size_t>(i + j * dims[u])] = FaceMaskCell{};
                }

                std::uint32_t positiveMask = solidMaskA & ~solidMaskB;
                std::uint32_t negativeMask = solidMaskB & ~solidMaskA;
                if (slice < 0) {
                    positiveMask = 0;
                }
                if ((slice + 1) >= dims[axis]) {
                    negativeMask = 0;
                }
                positiveMask &= kFaceRowBitMask;
                negativeMask &= kFaceRowBitMask;
                rowBits[static_cast<std::size_t>(j)] = positiveMask | negativeMask;

                std::uint32_t remainingRowBits = rowBits[static_cast<std::size_t>(j)];
                while (remainingRowBits != 0) {
                    const int i = CountTrailingZeros(remainingRowBits);
                    FaceMaskCell face{};
                    if ((positiveMask & (1u << i)) != 0) {
                        face = MakeFaceMaskCell(
                            GetBlockTextureLayer(
                                rowValuesA[static_cast<std::size_t>(i)],
                                GetBlockFaceForAxis(axis, true)
                            ),
                            true,
                            PackAoPattern(ComputeQuadAoPattern(sampleBlock, axis, true, slice + 1, i, j, 1, 1))
                        );
                    } else if ((negativeMask & (1u << i)) != 0) {
                        face = MakeFaceMaskCell(
                            GetBlockTextureLayer(
                                rowValuesB[static_cast<std::size_t>(i)],
                                GetBlockFaceForAxis(axis, false)
                            ),
                            false,
                            PackAoPattern(ComputeQuadAoPattern(sampleBlock, axis, false, slice + 1, i, j, 1, 1))
                        );
                    }
                    mask[static_cast<std::size_t>(i + j * dims[u])] = face;
                    remainingRowBits &= (remainingRowBits - 1);
                }
            }

            for (int j = 0; j < dims[v]; ++j) {
                while (rowBits[static_cast<std::size_t>(j)] != 0) {
                    const int i = CountTrailingZeros(rowBits[static_cast<std::size_t>(j)]);
                    const FaceMaskCell current = mask[static_cast<std::size_t>(i + j * dims[u])];

                    int width = 1;
                    while (i + width < dims[u] &&
                           (rowBits[static_cast<std::size_t>(j)] & (1u << (i + width))) != 0 &&
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

                    emitGreedyQuad(
                        axis,
                        IsPositiveFaceMaskCell(current),
                        slice + 1,
                        i,
                        j,
                        width,
                        height,
                        static_cast<int>(GetFaceMaskCellMaterialId(current))
                    );

                    const std::uint32_t rectMask = ((1u << width) - 1u) << i;
                    for (int row = 0; row < height; ++row) {
                        rowBits[static_cast<std::size_t>(j + row)] &= ~rectMask;
                        for (int col = 0; col < width; ++col) {
                            mask[static_cast<std::size_t>(i + col + (j + row) * dims[u])] = FaceMaskCell{};
                        }
                    }
                }
            }
        }
    }

    const auto profileEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gMeshBuildProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(profileEndTime - profileStartTime).count()
        )
    );
}

}  // namespace

VoxelWorld::VoxelWorld() = default;
VoxelWorld::~VoxelWorld() = default;
VoxelWorld::VoxelWorld(VoxelWorld&& other) noexcept
    : initialized_(other.initialized_),
      terrainConfigLoaded_(other.terrainConfigLoaded_),
      saveDirty_(other.saveDirty_),
      streamingWindowInitialized_(other.streamingWindowInitialized_),
      streamingCenterChunkX_(other.streamingCenterChunkX_),
      streamingCenterChunkZ_(other.streamingCenterChunkZ_),
      streamingKeepRadius_(other.streamingKeepRadius_),
      terrainConfig_(std::move(other.terrainConfig_)),
      chunkColumns_(std::move(other.chunkColumns_)),
      chunkTaskStates_(std::move(other.chunkTaskStates_)),
      desiredChunks_(std::move(other.desiredChunks_)),
      pendingChunkLoadQueue_(std::move(other.pendingChunkLoadQueue_)),
      dirtySubChunkQueue_(std::move(other.dirtySubChunkQueue_)),
      pendingRenderChunkUpdates_(std::move(other.pendingRenderChunkUpdates_)),
      pendingRenderChunkRemovals_(std::move(other.pendingRenderChunkRemovals_)),
      renderStatsDirty_(other.renderStatsDirty_),
      regionIndexCache_(std::move(other.regionIndexCache_)) {}

VoxelWorld& VoxelWorld::operator=(VoxelWorld&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    initialized_ = other.initialized_;
    terrainConfigLoaded_ = other.terrainConfigLoaded_;
    saveDirty_ = other.saveDirty_;
    streamingWindowInitialized_ = other.streamingWindowInitialized_;
    streamingCenterChunkX_ = other.streamingCenterChunkX_;
    streamingCenterChunkZ_ = other.streamingCenterChunkZ_;
    streamingKeepRadius_ = other.streamingKeepRadius_;
    terrainConfig_ = std::move(other.terrainConfig_);
    chunkColumns_ = std::move(other.chunkColumns_);
    chunkTaskStates_ = std::move(other.chunkTaskStates_);
    desiredChunks_ = std::move(other.desiredChunks_);
    pendingChunkLoadQueue_ = std::move(other.pendingChunkLoadQueue_);
    dirtySubChunkQueue_ = std::move(other.dirtySubChunkQueue_);
    pendingRenderChunkUpdates_ = std::move(other.pendingRenderChunkUpdates_);
    pendingRenderChunkRemovals_ = std::move(other.pendingRenderChunkRemovals_);
    renderStatsDirty_ = other.renderStatsDirty_;
    regionIndexCache_ = std::move(other.regionIndexCache_);
    return *this;
}

VoxelWorldRuntimeProfileSnapshot ConsumeVoxelWorldRuntimeProfileSnapshot() {
    VoxelWorldRuntimeProfileSnapshot snapshot{};
    snapshot.chunkLoad = ConsumeRuntimeProfileStage(gChunkLoadProfile);
    snapshot.diskLoad = ConsumeRuntimeProfileStage(gDiskLoadProfile);
    snapshot.generate = ConsumeRuntimeProfileStage(gGenerateProfile);
    snapshot.meshBuild = ConsumeRuntimeProfileStage(gMeshBuildProfile);
    snapshot.save = ConsumeRuntimeProfileStage(gSaveProfile);
    snapshot.unload = ConsumeRuntimeProfileStage(gUnloadProfile);
    snapshot.unloadCount = ConsumeRuntimeCountStage(gUnloadCountProfile);
    snapshot.saveFile = ConsumeRuntimeProfileStage(gSaveFileProfile);
    snapshot.saveCount = ConsumeRuntimeCountStage(gSaveCountProfile);
    snapshot.getBlock = ConsumeRuntimeProfileStage(gGetBlockProfile);
    snapshot.generatedBlock = ConsumeRuntimeProfileStage(gGeneratedBlockProfile);
    return snapshot;
}

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

std::uint32_t VoxelWorld::GetSubChunkBit(int subChunkIndex) {
    return 1u << static_cast<std::uint32_t>(subChunkIndex);
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

std::uint16_t VoxelWorld::SampleGeneratedBlock(int worldX, int worldY, int worldZ) const {
    const auto profileStartTime = Clock::now();
    const std::uint16_t block = WorldGenerator::SampleBlock(worldX, worldY, worldZ, terrainConfig_);
    const auto profileEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gGeneratedBlockProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(profileEndTime - profileStartTime).count()
        )
    );
    return block;
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
        ChunkTaskState* state = FindChunkTaskState(pendingChunk.chunkX, pendingChunk.chunkZ);
        if (state == nullptr) {
            continue;
        }
        state->loadQueued = false;

        if (!state->desired || state->resident || state->loadInFlight) {
            MaybeCleanupChunkTaskState(pendingChunk.chunkX, pendingChunk.chunkZ);
            continue;
        }

        state->loadInFlight = true;
        requests.push_back(pendingChunk);
    }

    return requests;
}

PreparedChunkColumn VoxelWorld::PrepareChunkColumn(int chunkX, int chunkZ) const {
    const auto profileStartTime = Clock::now();
    PreparedChunkColumn prepared{};
    prepared.id = {chunkX, chunkZ};

    if (TryTakePrefetchedChunkColumn(chunkX, chunkZ, prepared.column)) {
        prepared.generated = prepared.column.modified;
        const auto profileEndTime = Clock::now();
        RecordRuntimeProfileSample(
            gChunkLoadProfile,
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(profileEndTime - profileStartTime).count()
            )
        );
        return prepared;
    }

    const auto diskLoadStartTime = Clock::now();
    const bool loadedFromDisk = LoadChunkColumn(chunkX, chunkZ, prepared.column);
    const auto diskLoadEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gDiskLoadProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(diskLoadEndTime - diskLoadStartTime).count()
        )
    );

    if (!loadedFromDisk) {
        const auto generateStartTime = Clock::now();
        GenerateChunkColumn(chunkX, chunkZ, prepared.column);
        const auto generateEndTime = Clock::now();
        RecordRuntimeProfileSample(
            gGenerateProfile,
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(generateEndTime - generateStartTime).count()
            )
        );
        prepared.column.modified = true;
        prepared.generated = true;
    }

    const auto profileEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gChunkLoadProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(profileEndTime - profileStartTime).count()
        )
    );
    return prepared;
}

bool VoxelWorld::CommitPreparedChunkColumn(PreparedChunkColumn&& prepared) {
    EnsureInitialized();

    ChunkTaskState& state = GetOrCreateChunkTaskState(prepared.id.chunkX, prepared.id.chunkZ);
    state.loadInFlight = false;

    const std::int64_t chunkKey = MakeChunkKey(prepared.id.chunkX, prepared.id.chunkZ);
    if (chunkColumns_.contains(chunkKey)) {
        return false;
    }

    if (prepared.generated) {
        saveDirty_ = true;
    }

    auto [it, inserted] = chunkColumns_.emplace(
        chunkKey,
        std::make_shared<ChunkColumnData>(std::move(prepared.column))
    );
    (void)inserted;
    state.resident = true;
    state.retireRequested = !state.desired;

    renderStatsDirty_ = true;
    if (state.desired) {
        EnqueueAllDirtySubChunks(prepared.id.chunkX, prepared.id.chunkZ, *it->second);
        if (!HasPendingMeshWorkForChunk(prepared.id.chunkX, prepared.id.chunkZ)) {
            QueueRenderChunkUpdate(prepared.id.chunkX, prepared.id.chunkZ);
        }
    }
    return true;
}

std::vector<DirtySubChunkId> VoxelWorld::AcquireDirtyMeshRequestCandidates(
    std::size_t maxCount,
    int centerChunkX,
    int centerChunkZ,
    int keepRadius
) {
    EnsureInitialized();

    std::vector<DirtySubChunkId> candidates;
    candidates.reserve(maxCount);

    const std::size_t initialQueueCount = dirtySubChunkQueue_.size();
    for (std::size_t iteration = 0; iteration < initialQueueCount && candidates.size() < maxCount; ++iteration) {
        if (dirtySubChunkQueue_.empty()) {
            break;
        }

        const DirtySubChunkId id = dirtySubChunkQueue_.front();
        dirtySubChunkQueue_.pop_front();
        ChunkTaskState& state = GetOrCreateChunkTaskState(id.chunkX, id.chunkZ);
        const std::uint32_t dirtyBit = GetSubChunkBit(id.subChunkIndex);
        state.dirtyQueuedMask &= ~dirtyBit;

        const auto requeueIfStillRelevant = [&]() {
            if (state.desired || state.loadInFlight || (state.resident && !state.retireRequested)) {
                EnqueueDirtySubChunk(id.chunkX, id.chunkZ, id.subChunkIndex);
            }
        };

        const int dx = id.chunkX - centerChunkX;
        const int dz = id.chunkZ - centerChunkZ;
        if (dx * dx + dz * dz > keepRadius * keepRadius) {
            requeueIfStillRelevant();
            continue;
        }

        if ((state.dirtyInFlightMask & dirtyBit) != 0) {
            requeueIfStillRelevant();
            continue;
        }

        candidates.push_back(id);
    }

    return candidates;
}

DirtyMeshRequestHandleSelection VoxelWorld::ResolveDirtyMeshRequestHandles(const std::vector<DirtySubChunkId>& candidates) const {
    DirtyMeshRequestHandleSelection selection{};
    selection.readyHandles.reserve(candidates.size());
    selection.retryIds.reserve(candidates.size());

    for (const DirtySubChunkId& id : candidates) {
        if (id.subChunkIndex < 0 || id.subChunkIndex >= kSubChunkCountY) {
            continue;
        }

        std::shared_ptr<const ChunkColumnData> columnHandle = FindChunkColumnHandle(id.chunkX, id.chunkZ);
        if (!columnHandle) {
            selection.retryIds.push_back(id);
            continue;
        }
        if (columnHandle->subChunkMeshes.size() != kSubChunkCountY ||
            columnHandle->subChunks.size() != kSubChunkCountY) {
            selection.retryIds.push_back(id);
            continue;
        }
        if (!columnHandle->subChunkMeshes[static_cast<std::size_t>(id.subChunkIndex)].dirty) {
            continue;
        }

        selection.readyHandles.push_back({id, std::move(columnHandle)});
    }

    return selection;
}

void VoxelWorld::PrefetchChunkColumnsForMeshRequests(const DirtyMeshRequestHandleSelection& handleSelection) {
    EnsureInitialized();

    std::vector<PendingChunkId> missingChunks;
    missingChunks.reserve(handleSelection.readyHandles.size() * 9);

    {
        std::lock_guard lock(prefetchedChunkMutex_);
        for (const DirtyMeshRequestHandle& handle : handleSelection.readyHandles) {
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int chunkX = handle.id.chunkX + dx;
                    const int chunkZ = handle.id.chunkZ + dz;
                    const std::int64_t chunkKey = MakeChunkKey(chunkX, chunkZ);
                    if (chunkColumns_.contains(chunkKey) || prefetchedChunkColumns_.contains(chunkKey)) {
                        continue;
                    }
                    missingChunks.push_back({chunkX, chunkZ});
                }
            }
        }
    }

    std::sort(missingChunks.begin(), missingChunks.end(), [](const PendingChunkId& lhs, const PendingChunkId& rhs) {
        if (lhs.chunkZ != rhs.chunkZ) {
            return lhs.chunkZ < rhs.chunkZ;
        }
        return lhs.chunkX < rhs.chunkX;
    });
    missingChunks.erase(std::unique(missingChunks.begin(), missingChunks.end()), missingChunks.end());

    for (const PendingChunkId& chunkId : missingChunks) {
        ChunkColumnData column{};
        const bool loadedFromDisk = LoadChunkColumn(chunkId.chunkX, chunkId.chunkZ, column);
        if (!loadedFromDisk) {
            GenerateChunkColumn(chunkId.chunkX, chunkId.chunkZ, column);
            column.modified = true;
        }
        CachePrefetchedChunkColumn(chunkId.chunkX, chunkId.chunkZ, std::move(column));
    }
}

DirtyMeshRequestSelection VoxelWorld::ResolveDirtyMeshRequests(const DirtyMeshRequestHandleSelection& handleSelection) const {
    DirtyMeshRequestSelection selection{};
    selection.requests.reserve(handleSelection.readyHandles.size());
    selection.readyIds.reserve(handleSelection.readyHandles.size());
    selection.retryIds = handleSelection.retryIds;

    for (const DirtyMeshRequestHandle& handle : handleSelection.readyHandles) {
        const DirtySubChunkId& id = handle.id;
        const ChunkColumnData& column = *handle.column;
        if (column.subChunkMeshes.size() != kSubChunkCountY ||
            column.subChunks.size() != kSubChunkCountY) {
            selection.retryIds.push_back(id);
            continue;
        }
        if (!column.subChunkMeshes[static_cast<std::size_t>(id.subChunkIndex)].dirty) {
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

        selection.readyIds.push_back(id);
        selection.requests.push_back(std::move(input));
    }

    return selection;
}

void VoxelWorld::FinalizeDirtyMeshRequests(const DirtyMeshRequestSelection& selection) {
    EnsureInitialized();

    for (const DirtySubChunkId& id : selection.readyIds) {
        if (ChunkTaskState* state = FindChunkTaskState(id.chunkX, id.chunkZ)) {
            state->dirtyInFlightMask |= GetSubChunkBit(id.subChunkIndex);
        }
    }

    for (const DirtySubChunkId& id : selection.retryIds) {
        ChunkTaskState* state = FindChunkTaskState(id.chunkX, id.chunkZ);
        if (state == nullptr) {
            continue;
        }

        if (state->desired || state->loadInFlight || (state->resident && !state->retireRequested)) {
            EnqueueDirtySubChunk(id.chunkX, id.chunkZ, id.subChunkIndex);
        }
    }
}

std::vector<MeshBuildInput> VoxelWorld::AcquireDirtyMeshRequests(std::size_t maxCount, int centerChunkX, int centerChunkZ, int keepRadius) {
    EnsureInitialized();

    const std::vector<DirtySubChunkId> candidates =
        AcquireDirtyMeshRequestCandidates(maxCount, centerChunkX, centerChunkZ, keepRadius);
    DirtyMeshRequestHandleSelection handleSelection = ResolveDirtyMeshRequestHandles(candidates);
    DirtyMeshRequestSelection selection = ResolveDirtyMeshRequests(handleSelection);
    FinalizeDirtyMeshRequests(selection);

    return std::move(selection.requests);
}

PreparedSubChunkMesh VoxelWorld::PrepareSubChunkMesh(const MeshBuildInput& input) const {
    PreparedSubChunkMesh preparedMesh{};
    preparedMesh.id = input.id;

    const auto sampleBlock = [&input](int localX, int localY, int localZ) -> std::uint16_t {
        return input.blocks[GetMeshBuildSampleIndex(localX + 1, localY + 1, localZ + 1)];
    };
    BuildSubChunkQuadRecords(
        sampleBlock,
        input.minWorldX,
        input.minWorldY,
        input.minWorldZ,
        input.id.subChunkIndex,
        preparedMesh.quads
    );

    return preparedMesh;
}

bool VoxelWorld::CommitPreparedSubChunkMesh(PreparedSubChunkMesh&& preparedMesh) {
    const DirtySubChunkId id = preparedMesh.id;
    ChunkTaskState* state = FindChunkTaskState(id.chunkX, id.chunkZ);
    const std::uint32_t dirtyBit = GetSubChunkBit(id.subChunkIndex);
    if (state != nullptr) {
        state->dirtyInFlightMask &= ~dirtyBit;
    }

    std::shared_ptr<ChunkColumnData> columnHandle = FindChunkColumnHandle(id.chunkX, id.chunkZ);
    if (!columnHandle) {
        MaybeCleanupChunkTaskState(id.chunkX, id.chunkZ);
        return false;
    }
    if (id.subChunkIndex < 0 || id.subChunkIndex >= kSubChunkCountY) {
        return false;
    }
    if (columnHandle->subChunkMeshes.size() != kSubChunkCountY) {
        return false;
    }

    SubChunkMeshData& mesh = columnHandle->subChunkMeshes[static_cast<std::size_t>(id.subChunkIndex)];
    mesh.quads = std::move(preparedMesh.quads);
    mesh.dirty = state != nullptr && (state->dirtyQueuedMask & dirtyBit) != 0;
    ++mesh.revision;
    QueueRenderChunkUpdate(id.chunkX, id.chunkZ);
    MaybeCleanupChunkTaskState(id.chunkX, id.chunkZ);
    return true;
}

PendingRenderDrainSelection VoxelWorld::AcquirePendingRenderDrainSelection() {
    EnsureInitialized();

    PendingRenderDrainSelection selection{};
    selection.removals.reserve(pendingRenderChunkRemovals_.size());
    while (!pendingRenderChunkRemovals_.empty()) {
        selection.removals.push_back(pendingRenderChunkRemovals_.front());
        pendingRenderChunkRemovals_.pop_front();
    }

    selection.updates.reserve(pendingRenderChunkUpdates_.size());
    while (!pendingRenderChunkUpdates_.empty()) {
        selection.updates.push_back(pendingRenderChunkUpdates_.front());
        pendingRenderChunkUpdates_.pop_front();
    }

    return selection;
}

PendingRenderDrainHandleSelection VoxelWorld::ResolvePendingRenderDrainHandles(const PendingRenderDrainSelection& selection) const {
    PendingRenderDrainHandleSelection resolved{};
    resolved.loadedChunkCount = chunkColumns_.size();
    resolved.removals = selection.removals;
    resolved.updates.reserve(selection.updates.size());
    resolved.missingUpdates.reserve(selection.updates.size());

    for (const PendingChunkId& chunkId : selection.updates) {
        std::shared_ptr<const ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkId.chunkX, chunkId.chunkZ);
        if (!columnHandle) {
            resolved.missingUpdates.push_back(chunkId);
            continue;
        }
        resolved.updates.push_back({chunkId, std::move(columnHandle)});
    }

    return resolved;
}

ResolvedRenderDrainSelection VoxelWorld::ResolvePendingRenderDrainSelection(const PendingRenderDrainHandleSelection& selection) const {
    ResolvedRenderDrainSelection resolved{};
    resolved.loadedChunkCount = selection.loadedChunkCount;
    resolved.uploads.reserve(selection.updates.size());
    resolved.failedUpdates = selection.missingUpdates;

    for (const PendingRenderDrainHandle& handle : selection.updates) {
        const PendingChunkId& chunkId = handle.id;
        ChunkMeshBatchData batch{};
        if (CopyChunkMeshBatch(chunkId.chunkX, chunkId.chunkZ, batch)) {
            resolved.uploads.push_back(std::move(batch));
        } else {
            resolved.failedUpdates.push_back(chunkId);
        }
    }

    return resolved;
}

WorldRenderUpdate VoxelWorld::FinalizePendingRenderDrainSelection(
    const PendingRenderDrainSelection& selection,
    ResolvedRenderDrainSelection&& resolvedSelection
) {
    EnsureInitialized();

    WorldRenderUpdate update{};
    update.loadedChunkCount = resolvedSelection.loadedChunkCount;

    update.removals.reserve(selection.removals.size());
    for (const PendingChunkId& chunkId : selection.removals) {
        ChunkTaskState* state = FindChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
        if (state == nullptr || !state->renderRemovalQueued) {
            MaybeCleanupChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
            continue;
        }

        state->renderRemovalQueued = false;
        update.removals.push_back(chunkId);
        MaybeCleanupChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
    }

    update.uploads.reserve(resolvedSelection.uploads.size());
    for (ChunkMeshBatchData& batch : resolvedSelection.uploads) {
        const PendingChunkId chunkId = batch.id;
        ChunkTaskState* state = FindChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
        if (state == nullptr || !state->renderUpdateQueued) {
            MaybeCleanupChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
            continue;
        }

        state->renderUpdateQueued = false;
        update.uploads.push_back(std::move(batch));
        MaybeCleanupChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
    }

    for (const PendingChunkId& chunkId : resolvedSelection.failedUpdates) {
        ChunkTaskState* state = FindChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
        if (state == nullptr || !state->renderUpdateQueued) {
            MaybeCleanupChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
            continue;
        }

        state->renderUpdateQueued = false;
        if (state->desired &&
            state->resident &&
            (state->renderActive || state->loadInFlight || state->dirtyQueuedMask != 0 || state->dirtyInFlightMask != 0)) {
            state->renderUpdateQueued = true;
            pendingRenderChunkUpdates_.push_back(chunkId);
        }
        MaybeCleanupChunkTaskState(chunkId.chunkX, chunkId.chunkZ);
    }
    renderStatsDirty_ = !pendingRenderChunkUpdates_.empty() || !pendingRenderChunkRemovals_.empty();

    return update;
}

WorldRenderUpdate VoxelWorld::DrainRenderUpdates() {
    PendingRenderDrainSelection selection = AcquirePendingRenderDrainSelection();
    PendingRenderDrainHandleSelection handleSelection = ResolvePendingRenderDrainHandles(selection);
    ResolvedRenderDrainSelection resolvedSelection = ResolvePendingRenderDrainSelection(handleSelection);
    return FinalizePendingRenderDrainSelection(selection, std::move(resolvedSelection));
}

bool VoxelWorld::HasPendingRenderUpdates() const {
    return renderStatsDirty_ || !pendingRenderChunkUpdates_.empty() || !pendingRenderChunkRemovals_.empty();
}

std::size_t VoxelWorld::CountRemainingStreamingWork() const {
    std::size_t inFlightLoadCount = 0;
    std::size_t inFlightDirtyCount = 0;
    std::size_t retiringCount = 0;
    for (const auto& [chunkKey, state] : chunkTaskStates_) {
        (void)chunkKey;
        if (state.loadInFlight) {
            ++inFlightLoadCount;
        }
        if (state.dirtyInFlightMask != 0) {
            inFlightDirtyCount += static_cast<std::size_t>(std::popcount(state.dirtyInFlightMask));
        }
        if (state.resident && state.retireRequested) {
            ++retiringCount;
        }
    }

    const std::size_t remainingMissingCount = pendingChunkLoadQueue_.size() + inFlightLoadCount;
    const std::size_t remainingDirtyCount = dirtySubChunkQueue_.size() + inFlightDirtyCount;
    return remainingMissingCount + remainingDirtyCount + retiringCount;
}

std::vector<PendingChunkId> VoxelWorld::AcquireRetiredChunkUnloadCandidates(std::size_t unloadBudget) {
    std::vector<PendingChunkId> unloads;
    unloads.reserve(chunkTaskStates_.size());
    for (const auto& [chunkKey, state] : chunkTaskStates_) {
        (void)chunkKey;
        if (unloads.size() >= unloadBudget) {
            break;
        }

        if (!state.resident || state.desired || !state.retireRequested || state.loadInFlight || state.dirtyInFlightMask != 0) {
            continue;
        }

        const std::uint64_t unsignedKey = static_cast<std::uint64_t>(chunkKey);
        unloads.push_back({
            static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey >> 32)),
            static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey))
        });
    }

    return unloads;
}

std::size_t VoxelWorld::FinalizeStreamingWindow(int centerChunkX, int centerChunkZ, int keepRadius, std::size_t unloadBudget) {
    (void)centerChunkX;
    (void)centerChunkZ;
    (void)keepRadius;

    std::vector<PendingChunkId> unloads = AcquireRetiredChunkUnloadCandidates(unloadBudget);
    unloads = ExecuteChunkUnloads(unloads);
    FinalizeUnloadedChunkStates(unloads);
    return CountRemainingStreamingWork();
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

    auto [it, inserted] = chunkColumns_.emplace(
        chunkKey,
        std::make_shared<ChunkColumnData>(std::move(column))
    );
    (void)inserted;
    ChunkTaskState& state = GetOrCreateChunkTaskState(chunkX, chunkZ);
    state.resident = true;
    state.desired = desiredChunks_.contains(PendingChunkId{chunkX, chunkZ});
    state.retireRequested = !state.desired;
    if (state.desired) {
        EnqueueAllDirtySubChunks(chunkX, chunkZ, *it->second);
    }
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

std::shared_ptr<ChunkColumnData> VoxelWorld::FindChunkColumnHandle(int chunkX, int chunkZ) {
    const auto it = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    return it == chunkColumns_.end() ? nullptr : it->second;
}

std::shared_ptr<const ChunkColumnData> VoxelWorld::FindChunkColumnHandle(int chunkX, int chunkZ) const {
    const auto it = chunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    return it == chunkColumns_.end() ? nullptr : it->second;
}

bool VoxelWorld::TryTakePrefetchedChunkColumn(int chunkX, int chunkZ, ChunkColumnData& outColumn) const {
    std::lock_guard lock(prefetchedChunkMutex_);
    const auto it = prefetchedChunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (it == prefetchedChunkColumns_.end()) {
        return false;
    }
    outColumn = std::move(it->second);
    prefetchedChunkColumns_.erase(it);
    return true;
}

bool VoxelWorld::TryGetPrefetchedChunkColumn(int chunkX, int chunkZ, ChunkColumnData& outColumn) const {
    std::lock_guard lock(prefetchedChunkMutex_);
    const auto it = prefetchedChunkColumns_.find(MakeChunkKey(chunkX, chunkZ));
    if (it == prefetchedChunkColumns_.end()) {
        return false;
    }
    outColumn = it->second;
    return true;
}

void VoxelWorld::CachePrefetchedChunkColumn(int chunkX, int chunkZ, ChunkColumnData&& column) const {
    std::lock_guard lock(prefetchedChunkMutex_);
    prefetchedChunkColumns_.try_emplace(MakeChunkKey(chunkX, chunkZ), std::move(column));
}

void VoxelWorld::PrunePrefetchedChunkColumns(int centerChunkX, int centerChunkZ, int keepRadius) {
    std::lock_guard lock(prefetchedChunkMutex_);
    for (auto it = prefetchedChunkColumns_.begin(); it != prefetchedChunkColumns_.end();) {
        const std::uint64_t unsignedKey = static_cast<std::uint64_t>(it->first);
        const int chunkX = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey >> 32));
        const int chunkZ = static_cast<std::int32_t>(static_cast<std::uint32_t>(unsignedKey));
        const int dx = chunkX - centerChunkX;
        const int dz = chunkZ - centerChunkZ;
        if (dx * dx + dz * dz > keepRadius * keepRadius) {
            it = prefetchedChunkColumns_.erase(it);
        } else {
            ++it;
        }
    }
}

ChunkTaskState& VoxelWorld::GetOrCreateChunkTaskState(int chunkX, int chunkZ) {
    return chunkTaskStates_[MakeChunkKey(chunkX, chunkZ)];
}

ChunkTaskState* VoxelWorld::FindChunkTaskState(int chunkX, int chunkZ) {
    const auto it = chunkTaskStates_.find(MakeChunkKey(chunkX, chunkZ));
    return it == chunkTaskStates_.end() ? nullptr : &it->second;
}

const ChunkTaskState* VoxelWorld::FindChunkTaskState(int chunkX, int chunkZ) const {
    const auto it = chunkTaskStates_.find(MakeChunkKey(chunkX, chunkZ));
    return it == chunkTaskStates_.end() ? nullptr : &it->second;
}

void VoxelWorld::MaybeCleanupChunkTaskState(int chunkX, int chunkZ) {
    const std::int64_t chunkKey = MakeChunkKey(chunkX, chunkZ);
    const auto it = chunkTaskStates_.find(chunkKey);
    if (it == chunkTaskStates_.end()) {
        return;
    }

    const ChunkTaskState& state = it->second;
    if (state.desired ||
        state.resident ||
        state.loadQueued ||
        state.loadInFlight ||
        state.retireRequested ||
        state.renderActive ||
        state.renderUpdateQueued ||
        state.renderRemovalQueued ||
        state.dirtyQueuedMask != 0 ||
        state.dirtyInFlightMask != 0) {
        return;
    }

    chunkTaskStates_.erase(it);
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
        subChunkMesh.quads.clear();
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

    for (const PendingChunkId& queuedChunk : pendingChunkLoadQueue_) {
        if (ChunkTaskState* state = FindChunkTaskState(queuedChunk.chunkX, queuedChunk.chunkZ)) {
            state->loadQueued = false;
        }
    }

    std::vector<PendingChunk> pendingChunks;
    pendingChunks.reserve(desiredChunks_.size());
    for (const PendingChunkId& id : desiredChunks_) {
        ChunkTaskState& state = GetOrCreateChunkTaskState(id.chunkX, id.chunkZ);
        state.desired = true;
        state.retireRequested = false;
        if (state.resident || state.loadInFlight) {
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
    for (const PendingChunk& pendingChunk : pendingChunks) {
        GetOrCreateChunkTaskState(pendingChunk.id.chunkX, pendingChunk.id.chunkZ).loadQueued = true;
        pendingChunkLoadQueue_.push_back(pendingChunk.id);
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
    PrunePrefetchedChunkColumns(centerChunkX, centerChunkZ, keepRadius + 1);

    const std::unordered_set<PendingChunkId, PendingChunkIdHash> previousDesiredChunks = desiredChunks_;
    const std::unordered_set<PendingChunkId, PendingChunkIdHash> newDesiredChunks =
        CollectDesiredChunks(centerChunkX, centerChunkZ, keepRadius);

    for (const PendingChunkId& id : previousDesiredChunks) {
        if (!newDesiredChunks.contains(id)) {
            ChunkTaskState& state = GetOrCreateChunkTaskState(id.chunkX, id.chunkZ);
            state.desired = false;
            state.retireRequested = state.resident;
            QueueRenderChunkRemoval(id.chunkX, id.chunkZ);
            if (!state.resident && !state.loadInFlight) {
                MaybeCleanupChunkTaskState(id.chunkX, id.chunkZ);
            }
        }
    }

    for (const PendingChunkId& id : newDesiredChunks) {
        ChunkTaskState& state = GetOrCreateChunkTaskState(id.chunkX, id.chunkZ);
        state.desired = true;
        state.retireRequested = false;
    }

    desiredChunks_ = newDesiredChunks;

    for (const PendingChunkId& id : desiredChunks_) {
        if (previousDesiredChunks.contains(id)) {
            continue;
        }

        const auto columnIt = chunkColumns_.find(MakeChunkKey(id.chunkX, id.chunkZ));
        if (columnIt != chunkColumns_.end()) {
            EnqueueAllDirtySubChunks(id.chunkX, id.chunkZ, *columnIt->second);
            QueueRenderUpdatesForChunk(id.chunkX, id.chunkZ, *columnIt->second);
        }
    }

    RebuildPendingChunkQueue(centerChunkX, centerChunkZ);
}

void VoxelWorld::EnqueueDirtySubChunk(int chunkX, int chunkZ, int subChunkIndex, bool prioritize) {
    if (subChunkIndex < 0 || subChunkIndex >= kSubChunkCountY) {
        return;
    }

    ChunkTaskState& state = GetOrCreateChunkTaskState(chunkX, chunkZ);
    const std::uint32_t dirtyBit = GetSubChunkBit(subChunkIndex);
    const DirtySubChunkId id{chunkX, chunkZ, subChunkIndex};
    if ((state.dirtyQueuedMask & dirtyBit) != 0) {
        if (prioritize) {
            dirtySubChunkQueue_.erase(
                std::remove(dirtySubChunkQueue_.begin(), dirtySubChunkQueue_.end(), id),
                dirtySubChunkQueue_.end()
            );
            dirtySubChunkQueue_.push_front(id);
        }
        return;
    }

    state.dirtyQueuedMask |= dirtyBit;
    if (prioritize) {
        dirtySubChunkQueue_.push_front(id);
    } else {
        dirtySubChunkQueue_.push_back(id);
    }
}

void VoxelWorld::EnqueueAllDirtySubChunks(int chunkX, int chunkZ, ChunkColumnData& column) {
    const ChunkTaskState* state = FindChunkTaskState(chunkX, chunkZ);
    const bool shouldQueueWork = state == nullptr || (state->desired && !state->retireRequested);

    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        if (shouldQueueWork) {
            for (int subChunkIndex = kSubChunkCountY - 1; subChunkIndex >= 0; --subChunkIndex) {
                EnqueueDirtySubChunk(chunkX, chunkZ, subChunkIndex);
            }
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

        if (shouldQueueWork) {
            EnqueueDirtySubChunk(chunkX, chunkZ, subChunkIndex);
        }
    }
}

void VoxelWorld::RemoveQueuedDirtySubChunksForChunk(int chunkX, int chunkZ) {
    if (ChunkTaskState* state = FindChunkTaskState(chunkX, chunkZ)) {
        state->dirtyQueuedMask = 0;
    }

    if (dirtySubChunkQueue_.empty()) {
        return;
    }

    std::deque<DirtySubChunkId> retainedQueue;
    while (!dirtySubChunkQueue_.empty()) {
        DirtySubChunkId id = dirtySubChunkQueue_.front();
        dirtySubChunkQueue_.pop_front();
        if (id.chunkX == chunkX && id.chunkZ == chunkZ) {
            continue;
        }

        retainedQueue.push_back(id);
    }

    dirtySubChunkQueue_ = std::move(retainedQueue);
}

void VoxelWorld::MarkSubChunkDirty(int chunkX, int chunkZ, int subChunkIndex, bool prioritize) {
    if (subChunkIndex < 0 || subChunkIndex >= kSubChunkCountY) {
        return;
    }

    std::shared_ptr<ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
    if (!columnHandle) {
        return;
    }

    if (columnHandle->subChunkMeshes.size() != kSubChunkCountY) {
        InitializeSubChunkMeshes(*columnHandle, true);
        EnqueueAllDirtySubChunks(chunkX, chunkZ, *columnHandle);
        return;
    }

    SubChunkMeshData& subChunkMesh = columnHandle->subChunkMeshes[static_cast<std::size_t>(subChunkIndex)];
    subChunkMesh.dirty = true;
    const ChunkTaskState* state = FindChunkTaskState(chunkX, chunkZ);
    if (state == nullptr || (state->desired && !state->retireRequested)) {
        EnqueueDirtySubChunk(chunkX, chunkZ, subChunkIndex, prioritize);
    }
}

bool VoxelWorld::HasPendingMeshWorkForChunk(int chunkX, int chunkZ) const {
    if (const ChunkTaskState* state = FindChunkTaskState(chunkX, chunkZ)) {
        if (state->dirtyQueuedMask != 0 || state->dirtyInFlightMask != 0) {
            return true;
        }
    }

    const std::shared_ptr<const ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
    if (!columnHandle || columnHandle->subChunkMeshes.size() != kSubChunkCountY) {
        return false;
    }

    for (const SubChunkMeshData& mesh : columnHandle->subChunkMeshes) {
        if (mesh.dirty) {
            return true;
        }
    }

    return false;
}

void VoxelWorld::QueueRenderChunkUpdate(int chunkX, int chunkZ) {
    const std::shared_ptr<const ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
    if (!columnHandle || columnHandle->subChunkMeshes.size() != kSubChunkCountY) {
        return;
    }

    const PendingChunkId chunkId{chunkX, chunkZ};
    ChunkTaskState& state = GetOrCreateChunkTaskState(chunkX, chunkZ);
    if (!state.desired || !state.resident) {
        return;
    }

    state.renderRemovalQueued = false;

    if (!HasRenderableMesh(*columnHandle)) {
        if (state.renderActive) {
            state.renderActive = false;
            state.renderUpdateQueued = false;
            if (!state.renderRemovalQueued) {
                state.renderRemovalQueued = true;
                pendingRenderChunkRemovals_.push_back(chunkId);
            }
        }
        return;
    }

    state.renderActive = true;
    if (!state.renderUpdateQueued) {
        state.renderUpdateQueued = true;
        pendingRenderChunkUpdates_.push_back(chunkId);
    }
}

void VoxelWorld::QueueRenderChunkRemoval(int chunkX, int chunkZ) {
    const PendingChunkId chunkId{chunkX, chunkZ};
    ChunkTaskState& state = GetOrCreateChunkTaskState(chunkX, chunkZ);
    state.renderUpdateQueued = false;
    if (state.renderActive && !state.renderRemovalQueued) {
        state.renderActive = false;
        state.renderRemovalQueued = true;
        pendingRenderChunkRemovals_.push_back(chunkId);
    }
}

void VoxelWorld::QueueRenderUpdatesForChunk(int chunkX, int chunkZ, const ChunkColumnData& column) {
    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        return;
    }

    const ChunkTaskState* state = FindChunkTaskState(chunkX, chunkZ);
    if (!HasPendingMeshWorkForChunk(chunkX, chunkZ) ||
        (state != nullptr && state->renderRemovalQueued)) {
        QueueRenderChunkUpdate(chunkX, chunkZ);
    }
}

void VoxelWorld::UnloadRetiredChunks(std::size_t unloadBudget) {
    const std::vector<PendingChunkId> unloads = AcquireRetiredChunkUnloadCandidates(unloadBudget);
    FinalizeUnloadedChunkStates(ExecuteChunkUnloads(unloads));
}

std::vector<PendingChunkId> VoxelWorld::ExecuteChunkUnloads(const std::vector<PendingChunkId>& unloadCandidates) {
    const auto unloadStartTime = Clock::now();
    struct PendingUnload {
        int chunkX = 0;
        int chunkZ = 0;
    };

    std::vector<PendingUnload> unloads;
    unloads.reserve(unloadCandidates.size());
    for (const PendingChunkId& unloadCandidate : unloadCandidates) {
        unloads.push_back({
            unloadCandidate.chunkX,
            unloadCandidate.chunkZ
        });
    }

    if (unloads.empty()) {
        return {};
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

    std::vector<PendingChunkId> unloadedChunks;
    unloadedChunks.reserve(unloads.size());
    std::size_t unloadedChunkCount = 0;
    for (auto& [regionKey, group] : unloadRegions) {
        (void)regionKey;
        RegionSaveTask saveTask{};
        saveTask.regionX = group.regionX;
        saveTask.regionZ = group.regionZ;

        for (const PendingUnload& unload : group.chunks) {
            std::shared_ptr<ChunkColumnData> columnHandle = FindChunkColumnHandle(unload.chunkX, unload.chunkZ);
            if (!columnHandle) {
                continue;
            }

            if (columnHandle->modified) {
                RegionSaveChunkSnapshot snapshot{};
                snapshot.id = {unload.chunkX, unload.chunkZ};
                snapshot.column = *columnHandle;
                {
                    std::lock_guard lock(pendingSaveMutex_);
                    snapshot.generation = nextPendingSaveGeneration_++;
                    pendingSavedChunkColumns_[MakeChunkKey(unload.chunkX, unload.chunkZ)] =
                        PendingSavedChunkState{snapshot.generation, snapshot.column};
                }
                saveTask.chunks.push_back(std::move(snapshot));
            }
        }

        if (!saveTask.chunks.empty()) {
            std::lock_guard lock(pendingSaveMutex_);
            pendingSaveTasks_.push_back(std::move(saveTask));
        }

        for (const PendingUnload& unload : group.chunks) {
            const std::int64_t chunkKey = MakeChunkKey(unload.chunkX, unload.chunkZ);
            std::shared_ptr<ChunkColumnData> columnHandle = FindChunkColumnHandle(unload.chunkX, unload.chunkZ);
            if (!columnHandle) {
                unloadedChunks.push_back({unload.chunkX, unload.chunkZ});
                continue;
            }

            if (columnHandle->modified) {
                columnHandle->modified = false;
            }
            chunkColumns_.erase(chunkKey);
            unloadedChunks.push_back({unload.chunkX, unload.chunkZ});
            ++unloadedChunkCount;
        }
    }

    const auto unloadEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gUnloadProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(unloadEndTime - unloadStartTime).count()
        )
    );
    if (unloadedChunkCount > 0) {
        RecordRuntimeCountSample(gUnloadCountProfile, static_cast<std::uint64_t>(unloadedChunkCount));
    }

    saveDirty_ = false;
    for (const auto& [key, column] : chunkColumns_) {
        (void)key;
        if (column->modified) {
            saveDirty_ = true;
            break;
        }
    }

    return unloadedChunks;
}

void VoxelWorld::FinalizeUnloadedChunkStates(const std::vector<PendingChunkId>& unloadedChunks) {
    if (unloadedChunks.empty()) {
        return;
    }

    for (const PendingChunkId& unloadedChunk : unloadedChunks) {
        QueueRenderChunkRemoval(unloadedChunk.chunkX, unloadedChunk.chunkZ);
        RemoveQueuedDirtySubChunksForChunk(unloadedChunk.chunkX, unloadedChunk.chunkZ);
        if (ChunkTaskState* state = FindChunkTaskState(unloadedChunk.chunkX, unloadedChunk.chunkZ)) {
            state->resident = false;
            state->retireRequested = false;
            state->renderActive = false;
            state->renderUpdateQueued = false;
            state->dirtyQueuedMask = 0;
            state->dirtyInFlightMask = 0;
        }
        MaybeCleanupChunkTaskState(unloadedChunk.chunkX, unloadedChunk.chunkZ);
        renderStatsDirty_ = true;
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
        {
            std::lock_guard lock(regionIndexCacheMutex_);
            regionIndexCache_.clear();
        }
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
        if (!column->modified) {
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
            std::shared_ptr<ChunkColumnData> columnHandle = FindChunkColumnHandle(dirtyChunk.chunkX, dirtyChunk.chunkZ);
            if (!columnHandle) {
                continue;
            }

            overrides[MakeChunkKey(dirtyChunk.chunkX, dirtyChunk.chunkZ)] = columnHandle.get();
            columnHandle->modified = false;
        }

        SaveRegionOverrides(regionGroup.regionX, regionGroup.regionZ, overrides);
    }

    saveDirty_ = false;
}

bool VoxelWorld::LoadChunkColumn(int chunkX, int chunkZ, ChunkColumnData& outColumn) const {
    const std::int64_t chunkKey = MakeChunkKey(chunkX, chunkZ);
    {
        std::lock_guard lock(pendingSaveMutex_);
        if (const auto pendingIt = pendingSavedChunkColumns_.find(chunkKey);
            pendingIt != pendingSavedChunkColumns_.end()) {
            outColumn = pendingIt->second.column;
            outColumn.modified = false;
            return true;
        }
    }

    std::lock_guard regionFileLock(regionFileIoMutex_);
    const int regionX = GetRegionCoord(chunkX);
    const int regionZ = GetRegionCoord(chunkZ);
    RegionChunkIndexMetadata indexMetadata{};
    if (!TryGetRegionChunkIndexMetadata(regionX, regionZ, chunkX, chunkZ, indexMetadata)) {
        return false;
    }

    const std::filesystem::path regionPath = GetRegionFilePath(regionX, regionZ);
    std::ifstream stream(regionPath, std::ios::binary);
    if (!stream.is_open()) {
        InvalidateRegionIndexCacheEntry(regionX, regionZ);
        return false;
    }

    stream.seekg(static_cast<std::streamoff>(indexMetadata.offset), std::ios::beg);
    if (!stream) {
        throw std::runtime_error("Region chunk offset is invalid.");
    }

    DeserializeChunkColumnPayload(stream, outColumn);
    if (!stream) {
        throw std::runtime_error("Region chunk payload ended unexpectedly.");
    }

    outColumn.modified = false;
    return true;
}

bool VoxelWorld::TryGetRegionChunkIndexMetadata(
    int regionX,
    int regionZ,
    int chunkX,
    int chunkZ,
    RegionChunkIndexMetadata& outMetadata
) const {
    const std::int64_t regionKey = MakeRegionKey(regionX, regionZ);
    {
        std::lock_guard lock(regionIndexCacheMutex_);
        auto cacheIt = regionIndexCache_.find(regionKey);
        if (cacheIt != regionIndexCache_.end()) {
            if (!cacheIt->second.regionExists) {
                return false;
            }
            const auto indexIt = cacheIt->second.chunkIndex.find(MakeChunkKey(chunkX, chunkZ));
            if (indexIt == cacheIt->second.chunkIndex.end()) {
                return false;
            }
            outMetadata = indexIt->second;
            return true;
        }
    }

    const std::filesystem::path regionPath = GetRegionFilePath(regionX, regionZ);
    RefreshRegionIndexCacheEntry(regionX, regionZ, regionPath);

    std::lock_guard lock(regionIndexCacheMutex_);
    auto cacheIt = regionIndexCache_.find(regionKey);
    if (cacheIt == regionIndexCache_.end() || !cacheIt->second.regionExists) {
        return false;
    }
    const auto indexIt = cacheIt->second.chunkIndex.find(MakeChunkKey(chunkX, chunkZ));
    if (indexIt == cacheIt->second.chunkIndex.end()) {
        return false;
    }
    outMetadata = indexIt->second;
    return true;
}

void VoxelWorld::RefreshRegionIndexCacheEntry(int regionX, int regionZ, const std::filesystem::path& regionPath) const {
    std::lock_guard regionFileLock(regionFileIoMutex_);
    std::ifstream stream(regionPath, std::ios::binary);
    if (!stream.is_open()) {
        RegionIndexCacheEntry missingEntry{};
        missingEntry.regionExists = false;
        std::lock_guard lock(regionIndexCacheMutex_);
        regionIndexCache_[MakeRegionKey(regionX, regionZ)] = std::move(missingEntry);
        return;
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

    RegionIndexCacheEntry cacheEntry{};
    cacheEntry.regionExists = true;
    cacheEntry.chunkIndex.reserve(chunkCount);
    for (std::uint32_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
        RegionChunkIndexEntry indexEntry{};
        ReadBinary(stream, indexEntry.chunkX);
        ReadBinary(stream, indexEntry.chunkZ);
        ReadBinary(stream, indexEntry.offset);
        ReadBinary(stream, indexEntry.size);
        if (!stream) {
            throw std::runtime_error("Region index is corrupted.");
        }

        cacheEntry.chunkIndex.emplace(
            MakeChunkKey(indexEntry.chunkX, indexEntry.chunkZ),
            RegionChunkIndexMetadata{
                indexEntry.chunkX,
                indexEntry.chunkZ,
                indexEntry.offset,
                indexEntry.size,
            }
        );
    }

    std::lock_guard lock(regionIndexCacheMutex_);
    regionIndexCache_[MakeRegionKey(regionX, regionZ)] = std::move(cacheEntry);
}

void VoxelWorld::InvalidateRegionIndexCacheEntry(int regionX, int regionZ) const {
    std::lock_guard lock(regionIndexCacheMutex_);
    regionIndexCache_.erase(MakeRegionKey(regionX, regionZ));
}

std::uint32_t VoxelWorld::CollectDirtySubChunkMask(ChunkColumnData& column) {
    std::uint32_t dirtyMask = 0;
    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        return dirtyMask;
    }

    for (int subChunkIndex = 0; subChunkIndex < kSubChunkCountY; ++subChunkIndex) {
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

        dirtyMask |= GetSubChunkBit(subChunkIndex);
    }

    return dirtyMask;
}

bool VoxelWorld::HasRenderableMesh(const ChunkColumnData& column) {
    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        return false;
    }

    for (const SubChunkMeshData& mesh : column.subChunkMeshes) {
        if (!mesh.quads.empty()) {
            return true;
        }
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

std::vector<RegionSaveTask> VoxelWorld::DrainPendingSaveTasks(std::size_t maxCount) {
    std::lock_guard lock(pendingSaveMutex_);
    std::vector<RegionSaveTask> tasks;
    tasks.reserve(std::min(maxCount, pendingSaveTasks_.size()));
    while (!pendingSaveTasks_.empty() && tasks.size() < maxCount) {
        tasks.push_back(std::move(pendingSaveTasks_.front()));
        pendingSaveTasks_.pop_front();
    }
    return tasks;
}

void VoxelWorld::CompletePendingSaveTask(const RegionSaveTask& completedTask) {
    std::lock_guard lock(pendingSaveMutex_);
    for (const RegionSaveChunkSnapshot& snapshot : completedTask.chunks) {
        const std::int64_t chunkKey = MakeChunkKey(snapshot.id.chunkX, snapshot.id.chunkZ);
        const auto pendingIt = pendingSavedChunkColumns_.find(chunkKey);
        if (pendingIt != pendingSavedChunkColumns_.end() &&
            pendingIt->second.generation == snapshot.generation) {
            pendingSavedChunkColumns_.erase(pendingIt);
        }
    }
}

void VoxelWorld::ExecuteSaveTask(const RegionSaveTask& task) const {
    const auto saveStartTime = Clock::now();
    std::unordered_map<std::int64_t, const ChunkColumnData*> overrides;
    overrides.reserve(task.chunks.size());
    for (const RegionSaveChunkSnapshot& snapshot : task.chunks) {
        overrides.emplace(MakeChunkKey(snapshot.id.chunkX, snapshot.id.chunkZ), &snapshot.column);
    }
    SaveRegionOverrides(task.regionX, task.regionZ, overrides);
    const auto saveEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gSaveProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(saveEndTime - saveStartTime).count()
        )
    );
    if (!task.chunks.empty()) {
        RecordRuntimeCountSample(gSaveCountProfile, static_cast<std::uint64_t>(task.chunks.size()));
    }
}

bool VoxelWorld::HasPendingSaveTasks() const {
    std::lock_guard lock(pendingSaveMutex_);
    return !pendingSaveTasks_.empty();
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
    const auto saveFileStartTime = Clock::now();
    std::lock_guard regionFileLock(regionFileIoMutex_);

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
    {
        std::ifstream existingStream(regionPath, std::ios::binary);
        if (existingStream.is_open()) {

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

    const auto saveFileEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gSaveFileProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(saveFileEndTime - saveFileStartTime).count()
        )
    );
}

    std::sort(serializedChunks.begin(), serializedChunks.end(), [](const SerializedRegionChunk& lhs, const SerializedRegionChunk& rhs) {
        if (lhs.chunkZ != rhs.chunkZ) {
            return lhs.chunkZ < rhs.chunkZ;
        }
        return lhs.chunkX < rhs.chunkX;
    });

    const std::filesystem::path tempRegionPath = regionPath.string() + ".tmp";
    {
        std::ofstream stream(tempRegionPath, std::ios::binary | std::ios::trunc);
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
        std::uint64_t currentOffset =
            headerSize + indexEntrySize * static_cast<std::uint64_t>(serializedChunks.size());

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
        stream.flush();
        stream.close();

        std::error_code replaceError;
        std::filesystem::rename(tempRegionPath, regionPath, replaceError);
        if (replaceError) {
            std::filesystem::remove(regionPath, replaceError);
            replaceError.clear();
            std::filesystem::rename(tempRegionPath, regionPath, replaceError);
            if (replaceError) {
                std::filesystem::remove(tempRegionPath, replaceError);
                throw std::runtime_error("Failed to replace region file.");
            }
        }
    }

    InvalidateRegionIndexCacheEntry(regionX, regionZ);
}

void VoxelWorld::LoadRegionFile(int regionX, int regionZ, std::unordered_map<std::int64_t, ChunkColumnData>& outColumns) const {
    outColumns.clear();

    const std::filesystem::path regionPath = GetRegionFilePath(regionX, regionZ);
    std::lock_guard regionFileLock(regionFileIoMutex_);
    std::ifstream stream(regionPath, std::ios::binary);
    if (!stream.is_open()) {
        return;
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

    std::lock_guard regionFileLock(regionFileIoMutex_);
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
    return static_cast<const VoxelWorld&>(*this).GetBlock(worldX, worldY, worldZ);
}

std::uint16_t VoxelWorld::GetBlock(int worldX, int worldY, int worldZ) const {
    const auto profileStartTime = Clock::now();
    if (worldY < 0 || worldY >= kWorldSizeY) {
        return 0;
    }

    const_cast<VoxelWorld*>(this)->EnsureInitialized();

    const int chunkX = FloorDiv(worldX, kChunkSizeX);
    const int chunkZ = FloorDiv(worldZ, kChunkSizeZ);
    const int localX = PositiveMod(worldX, kChunkSizeX);
    const int localZ = PositiveMod(worldZ, kChunkSizeZ);
    const int subChunkIndex = GetSubChunkIndex(worldY);
    const int localY = worldY - subChunkIndex * kSubChunkSize;

    const std::shared_ptr<const ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
    std::uint16_t blockValue = 0;
    if (!columnHandle || columnHandle->subChunks.size() != kSubChunkCountY) {
        ChunkColumnData prefetchedColumn{};
        if (TryGetPrefetchedChunkColumn(chunkX, chunkZ, prefetchedColumn) &&
            prefetchedColumn.subChunks.size() == kSubChunkCountY) {
            blockValue = GetSubChunkBlock(
                prefetchedColumn.subChunks[static_cast<std::size_t>(subChunkIndex)],
                localX,
                localY,
                localZ
            );
        } else {
            blockValue = SampleGeneratedBlock(worldX, worldY, worldZ);
        }
    } else {
        blockValue = GetSubChunkBlock(
            columnHandle->subChunks[static_cast<std::size_t>(subChunkIndex)],
            localX,
            localY,
            localZ
        );
    }

    const auto profileEndTime = Clock::now();
    RecordRuntimeProfileSample(
        gGetBlockProfile,
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(profileEndTime - profileStartTime).count()
        )
    );
    return blockValue;
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
    std::shared_ptr<ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
    if (!columnHandle) {
        return false;
    }

    if (columnHandle->subChunks.size() != kSubChunkCountY) {
        InitializeVoxelSubChunks(*columnHandle);
    }

    SubChunkVoxelData& subChunk = columnHandle->subChunks[static_cast<std::size_t>(subChunkIndex)];
    if (GetSubChunkBlock(subChunk, localX, localY, localZ) == blockValue) {
        return false;
    }

    SetSubChunkBlock(subChunk, localX, localY, localZ, blockValue);
    TryCollapseSubChunk(subChunk);

    columnHandle->modified = true;
    saveDirty_ = true;

    MarkSubChunkDirty(chunkX, chunkZ, subChunkIndex, true);
    if (worldY > 0 && worldY % kSubChunkSize == 0) {
        MarkSubChunkDirty(chunkX, chunkZ, subChunkIndex - 1, true);
    }
    if (worldY + 1 < kWorldSizeY && (worldY + 1) % kSubChunkSize == 0) {
        MarkSubChunkDirty(chunkX, chunkZ, subChunkIndex + 1, true);
    }

    if (localX == 0) {
        EnsureChunkColumn(chunkX - 1, chunkZ);
        MarkSubChunkDirty(chunkX - 1, chunkZ, subChunkIndex, true);
    }
    if (localX == kChunkSizeX - 1) {
        EnsureChunkColumn(chunkX + 1, chunkZ);
        MarkSubChunkDirty(chunkX + 1, chunkZ, subChunkIndex, true);
    }
    if (localZ == 0) {
        EnsureChunkColumn(chunkX, chunkZ - 1);
        MarkSubChunkDirty(chunkX, chunkZ - 1, subChunkIndex, true);
    }
    if (localZ == kChunkSizeZ - 1) {
        EnsureChunkColumn(chunkX, chunkZ + 1);
        MarkSubChunkDirty(chunkX, chunkZ + 1, subChunkIndex, true);
    }

    return true;
}

bool VoxelWorld::CopyChunkMeshBatch(int chunkX, int chunkZ, ChunkMeshBatchData& outBatch) const {
    const std::shared_ptr<const ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
    if (!columnHandle || columnHandle->subChunkMeshes.size() != kSubChunkCountY) {
        return false;
    }

    outBatch.id = {chunkX, chunkZ};
    outBatch.quads.clear();

    std::size_t totalQuadCount = 0;
    for (const SubChunkMeshData& subChunkMesh : columnHandle->subChunkMeshes) {
        totalQuadCount += subChunkMesh.quads.size();
    }

    if (totalQuadCount == 0) {
        return false;
    }

    outBatch.quads.reserve(totalQuadCount);

    for (const SubChunkMeshData& subChunkMesh : columnHandle->subChunkMeshes) {
        if (subChunkMesh.quads.empty()) {
            continue;
        }
        outBatch.quads.insert(outBatch.quads.end(), subChunkMesh.quads.begin(), subChunkMesh.quads.end());
    }

    return !outBatch.quads.empty();
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

            std::shared_ptr<const ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
            if (!columnHandle) {
                continue;
            }

            if (columnHandle->subChunkMeshes.size() != kSubChunkCountY) {
                continue;
            }

            for (int subChunkIndex = 0; subChunkIndex < kSubChunkCountY; ++subChunkIndex) {
                const SubChunkMeshData& subChunkMesh = columnHandle->subChunkMeshes[static_cast<std::size_t>(subChunkIndex)];
                if (subChunkMesh.quads.empty()) {
                    continue;
                }

                WorldVisibleBatch batch{};
                batch.id = {chunkX, chunkZ, subChunkIndex};
                batch.revision = subChunkMesh.revision;

                outMesh.totalVertexCount += static_cast<std::uint32_t>(subChunkMesh.quads.size() * 4);
                outMesh.totalIndexCount += static_cast<std::uint32_t>(subChunkMesh.quads.size() * 6);
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

void VoxelWorld::RebuildSubChunkMesh(int chunkX, int chunkZ, int subChunkIndex) {
    std::shared_ptr<ChunkColumnData> columnHandle = FindChunkColumnHandle(chunkX, chunkZ);
    if (!columnHandle) {
        return;
    }

    ChunkColumnData& column = *columnHandle;
    if (column.subChunkMeshes.size() != kSubChunkCountY) {
        InitializeSubChunkMeshes(column, true);
    }
    if (column.subChunks.size() != kSubChunkCountY) {
        InitializeVoxelSubChunks(column);
    }

    SubChunkMeshData& mesh = column.subChunkMeshes[static_cast<std::size_t>(subChunkIndex)];
    const SubChunkVoxelData& voxelSubChunk = column.subChunks[static_cast<std::size_t>(subChunkIndex)];
    mesh.quads.clear();

    const int minWorldY = subChunkIndex * kSubChunkSize;
    const int minWorldX = chunkX * kChunkSizeX;
    const int minWorldZ = chunkZ * kChunkSizeZ;

    if (voxelSubChunk.isUniform && voxelSubChunk.uniformBlock == 0) {
        mesh.dirty = false;
        ++mesh.revision;
        return;
    }

    const auto sampleBlock = [&](int localX, int localY, int localZ) -> std::uint16_t {
        if (localX >= 0 && localX < kChunkSizeX &&
            localY >= 0 && localY < kSubChunkSize &&
            localZ >= 0 && localZ < kChunkSizeZ) {
            return GetSubChunkBlock(voxelSubChunk, localX, localY, localZ);
        }

        return GetBlock(minWorldX + localX, minWorldY + localY, minWorldZ + localZ);
    };
    BuildSubChunkQuadRecords(sampleBlock, minWorldX, minWorldY, minWorldZ, subChunkIndex, mesh.quads);

    mesh.dirty = false;
    ++mesh.revision;
}
