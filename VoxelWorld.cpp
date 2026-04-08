#include "VoxelWorld.h"

#include <algorithm>

namespace {

void AppendQuad(
    std::vector<WorldVertex>& vertices,
    const Vec3& p0,
    const Vec3& p1,
    const Vec3& p2,
    const Vec3& p3
) {
    vertices.push_back({{p0.x, p0.y, p0.z}, {0.0f, 0.0f}});
    vertices.push_back({{p1.x, p1.y, p1.z}, {1.0f, 0.0f}});
    vertices.push_back({{p2.x, p2.y, p2.z}, {1.0f, 1.0f}});

    vertices.push_back({{p0.x, p0.y, p0.z}, {0.0f, 0.0f}});
    vertices.push_back({{p2.x, p2.y, p2.z}, {1.0f, 1.0f}});
    vertices.push_back({{p3.x, p3.y, p3.z}, {0.0f, 1.0f}});
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

void VoxelWorld::BuildVisibleMesh(
    int centerChunkX,
    int centerChunkZ,
    int renderRadius,
    std::vector<WorldVertex>& outVertices
) {
    outVertices.clear();

    const int minChunkX = std::max(0, centerChunkX - renderRadius);
    const int maxChunkX = std::min(kWorldChunkCountX - 1, centerChunkX + renderRadius);
    const int minChunkZ = std::max(0, centerChunkZ - renderRadius);
    const int maxChunkZ = std::min(kWorldChunkCountZ - 1, centerChunkZ + renderRadius);

    EnsureRange(minChunkX, maxChunkX, minChunkZ, maxChunkZ);

    for (int chunkZ = minChunkZ; chunkZ <= maxChunkZ; ++chunkZ) {
        for (int chunkX = minChunkX; chunkX <= maxChunkX; ++chunkX) {
            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ) {
                for (int localX = 0; localX < kChunkSizeX; ++localX) {
                    const int worldX = chunkX * kChunkSizeX + localX;
                    const int worldZ = chunkZ * kChunkSizeZ + localZ;
                    const int topBlockY = kGroundHeight - 1;

                    AppendTopFace(outVertices, worldX, topBlockY, worldZ);

                    if (worldX == 0) {
                        for (int worldY = 0; worldY < kGroundHeight; ++worldY) {
                            AppendWestFace(outVertices, worldX, worldY, worldZ);
                        }
                    }

                    if (worldX == kWorldSizeX - 1) {
                        for (int worldY = 0; worldY < kGroundHeight; ++worldY) {
                            AppendEastFace(outVertices, worldX, worldY, worldZ);
                        }
                    }

                    if (worldZ == 0) {
                        for (int worldY = 0; worldY < kGroundHeight; ++worldY) {
                            AppendNorthFace(outVertices, worldX, worldY, worldZ);
                        }
                    }

                    if (worldZ == kWorldSizeZ - 1) {
                        for (int worldY = 0; worldY < kGroundHeight; ++worldY) {
                            AppendSouthFace(outVertices, worldX, worldY, worldZ);
                        }
                    }
                }
            }
        }
    }
}

std::size_t VoxelWorld::GetLoadedChunkCount() const {
    return columns_.size();
}

void VoxelWorld::AppendTopFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const {
    const float x = static_cast<float>(worldX);
    const float y = static_cast<float>(worldY + 1);
    const float z = static_cast<float>(worldZ);

    AppendQuad(
        vertices,
        {x, y, z},
        {x + 1.0f, y, z},
        {x + 1.0f, y, z + 1.0f},
        {x, y, z + 1.0f}
    );
}

void VoxelWorld::AppendNorthFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const {
    const float x = static_cast<float>(worldX);
    const float y = static_cast<float>(worldY);
    const float z = static_cast<float>(worldZ);

    AppendQuad(
        vertices,
        {x + 1.0f, y, z},
        {x, y, z},
        {x, y + 1.0f, z},
        {x + 1.0f, y + 1.0f, z}
    );
}

void VoxelWorld::AppendSouthFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const {
    const float x = static_cast<float>(worldX);
    const float y = static_cast<float>(worldY);
    const float z = static_cast<float>(worldZ);

    AppendQuad(
        vertices,
        {x, y, z + 1.0f},
        {x + 1.0f, y, z + 1.0f},
        {x + 1.0f, y + 1.0f, z + 1.0f},
        {x, y + 1.0f, z + 1.0f}
    );
}

void VoxelWorld::AppendWestFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const {
    const float x = static_cast<float>(worldX);
    const float y = static_cast<float>(worldY);
    const float z = static_cast<float>(worldZ);

    AppendQuad(
        vertices,
        {x, y, z},
        {x, y, z + 1.0f},
        {x, y + 1.0f, z + 1.0f},
        {x, y + 1.0f, z}
    );
}

void VoxelWorld::AppendEastFace(std::vector<WorldVertex>& vertices, int worldX, int worldY, int worldZ) const {
    const float x = static_cast<float>(worldX + 1);
    const float y = static_cast<float>(worldY);
    const float z = static_cast<float>(worldZ);

    AppendQuad(
        vertices,
        {x, y, z + 1.0f},
        {x, y, z},
        {x, y + 1.0f, z},
        {x, y + 1.0f, z + 1.0f}
    );
}
