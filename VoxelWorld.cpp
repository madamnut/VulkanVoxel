#include "VoxelWorld.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kFrustumPaddingDegrees = 8.0f;

void AppendQuad(
    WorldMeshData& mesh,
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

}  // namespace

std::int64_t VoxelWorld::MakeChunkKey(int chunkX, int chunkZ) {
    return (static_cast<std::int64_t>(chunkX) << 32) |
           static_cast<std::uint32_t>(chunkZ);
}

int VoxelWorld::GetSubChunkIndex(int worldY) {
    return worldY / kSubChunkSize;
}

int VoxelWorld::GetSubChunkBlockIndex(int localX, int localY, int localZ) {
    return localY * kSubChunkSize * kSubChunkSize + localZ * kSubChunkSize + localX;
}

void VoxelWorld::EnsureChunkColumn(int chunkX, int chunkZ) {
    if (chunkX < 0 || chunkX >= kWorldChunkCountX || chunkZ < 0 || chunkZ >= kWorldChunkCountZ) {
        return;
    }

    const std::int64_t key = MakeChunkKey(chunkX, chunkZ);
    if (columns_.contains(key)) {
        return;
    }

    ChunkColumn column{};
    column.chunkX = chunkX;
    column.chunkZ = chunkZ;
    GenerateChunkColumn(column);
    columns_.emplace(key, std::move(column));
}

void VoxelWorld::EnsureRange(int minChunkX, int maxChunkX, int minChunkZ, int maxChunkZ) {
    for (int chunkZ = minChunkZ; chunkZ <= maxChunkZ; ++chunkZ) {
        for (int chunkX = minChunkX; chunkX <= maxChunkX; ++chunkX) {
            EnsureChunkColumn(chunkX, chunkZ);
        }
    }
}

void VoxelWorld::GenerateChunkColumn(ChunkColumn& column) const {
    for (int subChunkIndex = 0; subChunkIndex < kSubChunkCountY; ++subChunkIndex) {
        SubChunk& subChunk = column.subChunks[subChunkIndex];
        const int subChunkMinY = subChunkIndex * kSubChunkSize;
        const int subChunkMaxY = subChunkMinY + kSubChunkSize - 1;

        if (subChunkMaxY < kGroundHeight) {
            subChunk.blocks.fill(1);
            subChunk.isEmpty = false;
            subChunk.isFull = true;
            continue;
        }

        if (subChunkMinY >= kGroundHeight) {
            subChunk.blocks.fill(0);
            subChunk.isEmpty = true;
            subChunk.isFull = false;
            continue;
        }

        for (int localY = 0; localY < kSubChunkSize; ++localY) {
            const int worldY = subChunkMinY + localY;
            const std::uint8_t blockValue = worldY < kGroundHeight ? 1 : 0;

            for (int localZ = 0; localZ < kSubChunkSize; ++localZ) {
                for (int localX = 0; localX < kSubChunkSize; ++localX) {
                    subChunk.blocks[GetSubChunkBlockIndex(localX, localY, localZ)] = blockValue;
                }
            }
        }

        subChunk.isEmpty = false;
        subChunk.isFull = false;
    }
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
    const Mat4 projection = Perspective(paddedVerticalFov * 0.5f * 2.0f * kPi / 180.0f, aspectRatio, 0.1f, 2048.0f);
    const Mat4 view = LookAt(cameraPosition, cameraPosition + forward, {0.0f, 1.0f, 0.0f});
    const Mat4 viewProjection = Multiply(projection, view);

    const float minX = static_cast<float>(chunkX * kChunkSizeX);
    const float maxX = minX + static_cast<float>(kChunkSizeX);
    const float minY = 0.0f;
    const float maxY = static_cast<float>(kGroundHeight);
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
    outMesh.vertices.clear();
    outMesh.indices.clear();

    const int minChunkX = std::max(0, centerChunkX - renderRadius);
    const int maxChunkX = std::min(kWorldChunkCountX - 1, centerChunkX + renderRadius);
    const int minChunkZ = std::max(0, centerChunkZ - renderRadius);
    const int maxChunkZ = std::min(kWorldChunkCountZ - 1, centerChunkZ + renderRadius);

    EnsureRange(minChunkX, maxChunkX, minChunkZ, maxChunkZ);

    const int chunkCountX = maxChunkX - minChunkX + 1;
    const int chunkCountZ = maxChunkZ - minChunkZ + 1;
    std::vector<bool> visibleMask(static_cast<std::size_t>(chunkCountX * chunkCountZ), false);
    std::vector<bool> processed(static_cast<std::size_t>(chunkCountX * chunkCountZ), false);

    auto maskIndex = [chunkCountX](int x, int z) {
        return static_cast<std::size_t>(z * chunkCountX + x);
    };

    for (int chunkZ = minChunkZ; chunkZ <= maxChunkZ; ++chunkZ) {
        for (int chunkX = minChunkX; chunkX <= maxChunkX; ++chunkX) {
            const bool visible = IsChunkInsideFrustum(
                chunkX,
                chunkZ,
                cameraPosition,
                cameraForward,
                verticalFovDegrees,
                aspectRatio
            );
            visibleMask[maskIndex(chunkX - minChunkX, chunkZ - minChunkZ)] = visible;
        }
    }

    for (int localChunkZ = 0; localChunkZ < chunkCountZ; ++localChunkZ) {
        for (int localChunkX = 0; localChunkX < chunkCountX; ++localChunkX) {
            const std::size_t index = maskIndex(localChunkX, localChunkZ);
            if (!visibleMask[index] || processed[index]) {
                continue;
            }

            int width = 1;
            while (localChunkX + width < chunkCountX) {
                const std::size_t nextIndex = maskIndex(localChunkX + width, localChunkZ);
                if (!visibleMask[nextIndex] || processed[nextIndex]) {
                    break;
                }
                ++width;
            }

            int depth = 1;
            bool canGrow = true;
            while (localChunkZ + depth < chunkCountZ && canGrow) {
                for (int offsetX = 0; offsetX < width; ++offsetX) {
                    const std::size_t nextIndex = maskIndex(localChunkX + offsetX, localChunkZ + depth);
                    if (!visibleMask[nextIndex] || processed[nextIndex]) {
                        canGrow = false;
                        break;
                    }
                }
                if (canGrow) {
                    ++depth;
                }
            }

            for (int offsetZ = 0; offsetZ < depth; ++offsetZ) {
                for (int offsetX = 0; offsetX < width; ++offsetX) {
                    processed[maskIndex(localChunkX + offsetX, localChunkZ + offsetZ)] = true;
                }
            }

            AppendTopQuad(
                outMesh,
                (minChunkX + localChunkX) * kChunkSizeX,
                (minChunkZ + localChunkZ) * kChunkSizeZ,
                width * kChunkSizeX,
                depth * kChunkSizeZ
            );
        }
    }

    for (int localChunkX = 0; localChunkX < chunkCountX; ) {
        const int chunkX = minChunkX + localChunkX;
        if (chunkX != 0) {
            ++localChunkX;
            continue;
        }

        int startZ = -1;
        int depth = 0;
        for (int localChunkZ = 0; localChunkZ < chunkCountZ; ++localChunkZ) {
            if (visibleMask[maskIndex(localChunkX, localChunkZ)]) {
                if (startZ < 0) {
                    startZ = localChunkZ;
                }
                ++depth;
            } else if (startZ >= 0) {
                AppendWestQuad(outMesh, (minChunkZ + startZ) * kChunkSizeZ, depth * kChunkSizeZ);
                startZ = -1;
                depth = 0;
            }
        }
        if (startZ >= 0) {
            AppendWestQuad(outMesh, (minChunkZ + startZ) * kChunkSizeZ, depth * kChunkSizeZ);
        }

        ++localChunkX;
    }

    for (int localChunkX = 0; localChunkX < chunkCountX; ) {
        const int chunkX = minChunkX + localChunkX;
        if (chunkX != kWorldChunkCountX - 1) {
            ++localChunkX;
            continue;
        }

        int startZ = -1;
        int depth = 0;
        for (int localChunkZ = 0; localChunkZ < chunkCountZ; ++localChunkZ) {
            if (visibleMask[maskIndex(localChunkX, localChunkZ)]) {
                if (startZ < 0) {
                    startZ = localChunkZ;
                }
                ++depth;
            } else if (startZ >= 0) {
                AppendEastQuad(outMesh, (minChunkZ + startZ) * kChunkSizeZ, depth * kChunkSizeZ);
                startZ = -1;
                depth = 0;
            }
        }
        if (startZ >= 0) {
            AppendEastQuad(outMesh, (minChunkZ + startZ) * kChunkSizeZ, depth * kChunkSizeZ);
        }

        ++localChunkX;
    }

    for (int localChunkZ = 0; localChunkZ < chunkCountZ; ) {
        const int chunkZ = minChunkZ + localChunkZ;
        if (chunkZ != 0) {
            ++localChunkZ;
            continue;
        }

        int startX = -1;
        int width = 0;
        for (int localChunkX = 0; localChunkX < chunkCountX; ++localChunkX) {
            if (visibleMask[maskIndex(localChunkX, localChunkZ)]) {
                if (startX < 0) {
                    startX = localChunkX;
                }
                ++width;
            } else if (startX >= 0) {
                AppendNorthQuad(outMesh, (minChunkX + startX) * kChunkSizeX, width * kChunkSizeX);
                startX = -1;
                width = 0;
            }
        }
        if (startX >= 0) {
            AppendNorthQuad(outMesh, (minChunkX + startX) * kChunkSizeX, width * kChunkSizeX);
        }

        ++localChunkZ;
    }

    for (int localChunkZ = 0; localChunkZ < chunkCountZ; ) {
        const int chunkZ = minChunkZ + localChunkZ;
        if (chunkZ != kWorldChunkCountZ - 1) {
            ++localChunkZ;
            continue;
        }

        int startX = -1;
        int width = 0;
        for (int localChunkX = 0; localChunkX < chunkCountX; ++localChunkX) {
            if (visibleMask[maskIndex(localChunkX, localChunkZ)]) {
                if (startX < 0) {
                    startX = localChunkX;
                }
                ++width;
            } else if (startX >= 0) {
                AppendSouthQuad(outMesh, (minChunkX + startX) * kChunkSizeX, width * kChunkSizeX);
                startX = -1;
                width = 0;
            }
        }
        if (startX >= 0) {
            AppendSouthQuad(outMesh, (minChunkX + startX) * kChunkSizeX, width * kChunkSizeX);
        }

        ++localChunkZ;
    }

    outMesh.loadedChunkCount = columns_.size();
}

std::size_t VoxelWorld::GetLoadedChunkCount() const {
    return columns_.size();
}

void VoxelWorld::AppendTopQuad(WorldMeshData& mesh, int startX, int startZ, int width, int depth) const {
    const float x = static_cast<float>(startX);
    const float y = static_cast<float>(kGroundHeight);
    const float z = static_cast<float>(startZ);

    AppendQuad(
        mesh,
        {x, y, z},
        {x, y, z + static_cast<float>(depth)},
        {x + static_cast<float>(width), y, z + static_cast<float>(depth)},
        {x + static_cast<float>(width), y, z},
        static_cast<float>(depth),
        static_cast<float>(width)
    );
}

void VoxelWorld::AppendNorthQuad(WorldMeshData& mesh, int startX, int width) const {
    const float x = static_cast<float>(startX);
    const float z = 0.0f;

    AppendQuad(
        mesh,
        {x + static_cast<float>(width), 0.0f, z},
        {x, 0.0f, z},
        {x, static_cast<float>(kGroundHeight), z},
        {x + static_cast<float>(width), static_cast<float>(kGroundHeight), z},
        static_cast<float>(width),
        static_cast<float>(kGroundHeight)
    );
}

void VoxelWorld::AppendSouthQuad(WorldMeshData& mesh, int startX, int width) const {
    const float x = static_cast<float>(startX);
    const float z = static_cast<float>(kWorldSizeZ);

    AppendQuad(
        mesh,
        {x, 0.0f, z},
        {x + static_cast<float>(width), 0.0f, z},
        {x + static_cast<float>(width), static_cast<float>(kGroundHeight), z},
        {x, static_cast<float>(kGroundHeight), z},
        static_cast<float>(width),
        static_cast<float>(kGroundHeight)
    );
}

void VoxelWorld::AppendWestQuad(WorldMeshData& mesh, int startZ, int depth) const {
    const float x = 0.0f;
    const float z = static_cast<float>(startZ);

    AppendQuad(
        mesh,
        {x, 0.0f, z},
        {x, 0.0f, z + static_cast<float>(depth)},
        {x, static_cast<float>(kGroundHeight), z + static_cast<float>(depth)},
        {x, static_cast<float>(kGroundHeight), z},
        static_cast<float>(depth),
        static_cast<float>(kGroundHeight)
    );
}

void VoxelWorld::AppendEastQuad(WorldMeshData& mesh, int startZ, int depth) const {
    const float x = static_cast<float>(kWorldSizeX);
    const float z = static_cast<float>(startZ);

    AppendQuad(
        mesh,
        {x, 0.0f, z + static_cast<float>(depth)},
        {x, 0.0f, z},
        {x, static_cast<float>(kGroundHeight), z},
        {x, static_cast<float>(kGroundHeight), z + static_cast<float>(depth)},
        static_cast<float>(depth),
        static_cast<float>(kGroundHeight)
    );
}
