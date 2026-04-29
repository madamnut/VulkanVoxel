#include "world/ChunkMesher.h"

#include "core/Math.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace
{
struct FaceMaskCell
{
    bool filled = false;
    std::uint32_t textureLayer = 0;
    std::array<std::uint8_t, 4> ao{};

    bool operator==(const FaceMaskCell& other) const
    {
        return filled == other.filled && (!filled || (textureLayer == other.textureLayer && ao == other.ao));
    }
};

std::uint8_t vertexAo(bool side1, bool side2, bool corner)
{
    if (side1 && side2)
    {
        return 0;
    }

    return static_cast<std::uint8_t>(
        3 - static_cast<int>(side1) - static_cast<int>(side2) - static_cast<int>(corner));
}

float aoFactor(std::uint8_t ao)
{
    return 0.55f + 0.15f * static_cast<float>(ao);
}
}

ChunkMesher::ChunkMesher(
    const WorldGenerator& worldGenerator,
    const BlockRegistry& blockRegistry)
    : worldGenerator_(worldGenerator)
    , blockRegistry_(blockRegistry)
{
}

void ChunkMesher::setWaterTextureLayer(std::uint32_t textureLayer)
{
    waterTextureLayer_ = textureLayer;
}

const BlockDefinition* ChunkMesher::blockDefinitionForId(std::uint16_t blockId) const
{
    return blockRegistry_.definitionForId(blockId);
}

bool ChunkMesher::isSolidBlock(std::uint16_t blockId) const
{
    return blockRegistry_.isSolid(blockId);
}

std::uint32_t ChunkMesher::textureLayerForBlockFace(std::uint16_t blockId, BlockFace face) const
{
    const BlockDefinition* definition = blockDefinitionForId(blockId);
    if (definition == nullptr)
    {
        return 0;
    }
    return definition->textureLayers[static_cast<std::size_t>(face)];
}

SubchunkBuildResult ChunkMesher::buildSubchunkMesh(ChunkBuildRequest request) const
{
    const GeneratedChunkColumn column = worldGenerator_.generateChunkColumn(request.coord);
    return buildSubchunkMesh(request, column);
}

ChunkBuildResult ChunkMesher::buildChunkMesh(ChunkCoord coord, std::uint64_t generation) const
{
    ChunkVoxelData voxels = worldGenerator_.generateChunkVoxels(coord);
    return buildChunkMesh(coord, generation, voxels.blockIds, voxels.fluidIds, voxels.fluidAmounts);
}

ChunkBuildResult ChunkMesher::buildChunkMesh(
    ChunkCoord coord,
    std::uint64_t generation,
    const std::vector<std::uint16_t>& blockIds,
    const std::vector<std::uint8_t>& fluidIds,
    const std::vector<std::uint8_t>& fluidAmounts) const
{
    ChunkBuildResult result = buildChunkMesh(
        coord,
        generation,
        worldGenerator_.generateChunkColumn(coord, blockIds, fluidIds, fluidAmounts));
    result.blockIds = blockIds;
    result.fluidIds = fluidIds;
    result.fluidAmounts = fluidAmounts;
    return result;
}

ChunkBuildResult ChunkMesher::buildChunkMesh(
    ChunkCoord coord,
    std::uint64_t generation,
    const GeneratedChunkColumn& column) const
{
    ChunkBuildResult result{};
    result.coord = coord;
    result.subchunks.reserve(kSubchunksPerChunk);

    for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
    {
        SubchunkBuildResult subchunk = buildSubchunkMesh(
            {
                coord,
                subchunkY,
                0,
                generation,
            },
            column);
        if (subchunk.indices.empty())
        {
            continue;
        }

        SubchunkDraw draw{};
        draw.chunkX = coord.x;
        draw.chunkZ = coord.z;
        draw.subchunkY = subchunkY;
        draw.range.vertexCount = static_cast<std::uint32_t>(subchunk.vertices.size());
        draw.range.firstIndex = static_cast<std::uint32_t>(result.indices.size());
        draw.range.indexCount = static_cast<std::uint32_t>(subchunk.indices.size());
        draw.range.vertexOffset = static_cast<std::int32_t>(result.vertices.size());

        result.vertices.insert(result.vertices.end(), subchunk.vertices.begin(), subchunk.vertices.end());
        result.indices.insert(result.indices.end(), subchunk.indices.begin(), subchunk.indices.end());
        result.subchunks.push_back(draw);
    }

    return result;
}

SubchunkBuildResult ChunkMesher::buildSubchunkMesh(
    ChunkBuildRequest request,
    const GeneratedChunkColumn& column) const
{
        const ChunkCoord chunk = request.coord;
        std::array<std::vector<BlockVertex>, kSubchunkSize> verticesByLocalY;
        std::array<std::vector<std::uint32_t>, kSubchunkSize> indicesByLocalY;
        std::vector<BlockVertex> subchunkVertices;
        std::vector<std::uint32_t> subchunkIndices;
        std::vector<SubchunkDraw> subchunkDraws;
        subchunkDraws.reserve(1);

        const int chunkX = chunk.x;
        const int chunkZ = chunk.z;
        const int subchunkY = request.subchunkY;
        const int chunkBaseX = chunkX * kChunkSizeX;
        const int chunkBaseZ = chunkZ * kChunkSizeZ;
        const int subchunkMinY = subchunkY * kSubchunkSize;

        int nonAirBlockCount = 0;
        for (int localPaddedZ = 1; localPaddedZ <= kChunkSizeZ; ++localPaddedZ)
        {
            for (int localPaddedX = 1; localPaddedX <= kChunkSizeX; ++localPaddedX)
            {
                for (int localY = 0; localY < kSubchunkSize; ++localY)
                {
                    const int y = subchunkMinY + localY;
                    if (column.blockAt(localPaddedX, y, localPaddedZ) != kAirBlockId ||
                        (column.fluidIdAt(localPaddedX, y, localPaddedZ) != kNoFluidId &&
                            column.fluidAt(localPaddedX, y, localPaddedZ) > 0))
                    {
                        ++nonAirBlockCount;
                    }
                }
            }
        }

        if (nonAirBlockCount == 0)
        {
            return {
                chunk,
                subchunkY,
                request.generation,
                {},
                {},
            };
        }

        auto isSolid = [&](int x, int y, int z) -> bool
        {
            const int localPaddedX = x - chunkBaseX + 1;
            const int localPaddedZ = z - chunkBaseZ + 1;
            return isSolidBlock(column.blockAt(localPaddedX, y, localPaddedZ));
        };

        auto blockIdAt = [&](int x, int y, int z) -> std::uint16_t
        {
            const int localPaddedX = x - chunkBaseX + 1;
            const int localPaddedZ = z - chunkBaseZ + 1;
            return column.blockAt(localPaddedX, y, localPaddedZ);
        };

        auto fluidAmountAt = [&](int x, int y, int z) -> std::uint8_t
        {
            const int localPaddedX = x - chunkBaseX + 1;
            const int localPaddedZ = z - chunkBaseZ + 1;
            return column.fluidAt(localPaddedX, y, localPaddedZ);
        };

        auto fluidIdAt = [&](int x, int y, int z) -> std::uint8_t
        {
            const int localPaddedX = x - chunkBaseX + 1;
            const int localPaddedZ = z - chunkBaseZ + 1;
            return column.fluidIdAt(localPaddedX, y, localPaddedZ);
        };

        auto emitGreedyRectangles = [](
            int width,
            int height,
            auto cellAt,
            auto emitRectangle)
        {
            std::array<bool, kSubchunkSize * kSubchunkSize> consumed{};
            auto index = [](int u, int v) -> std::size_t
            {
                return static_cast<std::size_t>(v * kSubchunkSize + u);
            };

            for (int v = 0; v < height; ++v)
            {
                for (int u = 0; u < width; ++u)
                {
                    const FaceMaskCell baseCell = cellAt(u, v);
                    if (consumed[index(u, v)] || !baseCell.filled)
                    {
                        continue;
                    }

                    int rectWidth = 1;
                    while (u + rectWidth < width &&
                           !consumed[index(u + rectWidth, v)] &&
                           cellAt(u + rectWidth, v) == baseCell)
                    {
                        ++rectWidth;
                    }

                    int rectHeight = 1;
                    bool canGrow = true;
                    while (v + rectHeight < height && canGrow)
                    {
                        for (int du = 0; du < rectWidth; ++du)
                        {
                            if (consumed[index(u + du, v + rectHeight)] ||
                                !(cellAt(u + du, v + rectHeight) == baseCell))
                            {
                                canGrow = false;
                                break;
                            }
                        }

                        if (canGrow)
                        {
                            ++rectHeight;
                        }
                    }

                    for (int dv = 0; dv < rectHeight; ++dv)
                    {
                        for (int du = 0; du < rectWidth; ++du)
                        {
                            consumed[index(u + du, v + dv)] = true;
                        }
                    }

                    emitRectangle(u, v, rectWidth, rectHeight, baseCell);
                }
            }
        };

        auto addFace = [&](
            int subchunkLocalY,
            std::array<Vec3, 4> corners,
            std::array<std::array<float, 2>, 4> uvs,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            std::vector<BlockVertex>& vertices = verticesByLocalY[static_cast<std::size_t>(subchunkLocalY)];
            std::vector<std::uint32_t>& indices = indicesByLocalY[static_cast<std::size_t>(subchunkLocalY)];
            const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());
            const float layer = static_cast<float>(textureLayer);
            vertices.push_back({{corners[0].x, corners[0].y, corners[0].z}, {uvs[0][0], uvs[0][1]}, aoFactor(ao[0]), layer});
            vertices.push_back({{corners[1].x, corners[1].y, corners[1].z}, {uvs[1][0], uvs[1][1]}, aoFactor(ao[1]), layer});
            vertices.push_back({{corners[2].x, corners[2].y, corners[2].z}, {uvs[2][0], uvs[2][1]}, aoFactor(ao[2]), layer});
            vertices.push_back({{corners[3].x, corners[3].y, corners[3].z}, {uvs[3][0], uvs[3][1]}, aoFactor(ao[3]), layer});

            indices.push_back(baseIndex + 0);
            indices.push_back(baseIndex + 1);
            indices.push_back(baseIndex + 2);
            indices.push_back(baseIndex + 0);
            indices.push_back(baseIndex + 2);
            indices.push_back(baseIndex + 3);
        };

        auto addFluidFace = [&](
            int subchunkLocalY,
            std::array<Vec3, 4> corners,
            float width,
            float height)
        {
            addFace(
                subchunkLocalY,
                corners,
                {{
                    {{width, height}},
                    {{width, 0.0f}},
                    {{0.0f, 0.0f}},
                    {{0.0f, height}},
                }},
                waterTextureLayer_,
                {{3, 3, 3, 3}});
        };

        auto computeAo = [&](
            int side1X, int side1Y, int side1Z,
            int side2X, int side2Y, int side2Z,
            int cornerX, int cornerY, int cornerZ) -> std::uint8_t
        {
            return vertexAo(
                isSolid(side1X, side1Y, side1Z),
                isSolid(side2X, side2Y, side2Z),
                isSolid(cornerX, cornerY, cornerZ));
        };

        auto topFaceAo = [&](int x, int z, int faceY) -> std::array<std::uint8_t, 4>
        {
            return {{
                computeAo(x - 1, faceY, z, x, faceY, z - 1, x - 1, faceY, z - 1),
                computeAo(x - 1, faceY, z, x, faceY, z + 1, x - 1, faceY, z + 1),
                computeAo(x + 1, faceY, z, x, faceY, z + 1, x + 1, faceY, z + 1),
                computeAo(x + 1, faceY, z, x, faceY, z - 1, x + 1, faceY, z - 1),
            }};
        };

        auto bottomFaceAo = [&](int x, int z, int faceY) -> std::array<std::uint8_t, 4>
        {
            const int outsideY = faceY - 1;
            return {{
                computeAo(x - 1, outsideY, z, x, outsideY, z + 1, x - 1, outsideY, z + 1),
                computeAo(x - 1, outsideY, z, x, outsideY, z - 1, x - 1, outsideY, z - 1),
                computeAo(x + 1, outsideY, z, x, outsideY, z - 1, x + 1, outsideY, z - 1),
                computeAo(x + 1, outsideY, z, x, outsideY, z + 1, x + 1, outsideY, z + 1),
            }};
        };

        auto positiveXFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideX = x + 1;
            return {{
                computeAo(outsideX, y - 1, z, outsideX, y, z - 1, outsideX, y - 1, z - 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z - 1, outsideX, y + 1, z - 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z + 1, outsideX, y + 1, z + 1),
                computeAo(outsideX, y - 1, z, outsideX, y, z + 1, outsideX, y - 1, z + 1),
            }};
        };

        auto negativeXFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideX = x - 1;
            return {{
                computeAo(outsideX, y - 1, z, outsideX, y, z + 1, outsideX, y - 1, z + 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z + 1, outsideX, y + 1, z + 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z - 1, outsideX, y + 1, z - 1),
                computeAo(outsideX, y - 1, z, outsideX, y, z - 1, outsideX, y - 1, z - 1),
            }};
        };

        auto positiveZFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideZ = z + 1;
            return {{
                computeAo(x, y - 1, outsideZ, x + 1, y, outsideZ, x + 1, y - 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x + 1, y, outsideZ, x + 1, y + 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x - 1, y, outsideZ, x - 1, y + 1, outsideZ),
                computeAo(x, y - 1, outsideZ, x - 1, y, outsideZ, x - 1, y - 1, outsideZ),
            }};
        };

        auto negativeZFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideZ = z - 1;
            return {{
                computeAo(x, y - 1, outsideZ, x - 1, y, outsideZ, x - 1, y - 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x - 1, y, outsideZ, x - 1, y + 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x + 1, y, outsideZ, x + 1, y + 1, outsideZ),
                computeAo(x, y - 1, outsideZ, x + 1, y, outsideZ, x + 1, y - 1, outsideZ),
            }};
        };

        auto addTopFace = [&](
            int subchunkLocalY,
            int faceY,
            int x,
            int z,
            int width,
            int depth,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            const float fx = static_cast<float>(x);
            const float fz = static_cast<float>(z);
            const float xEnd = static_cast<float>(x + width);
            const float zEnd = static_cast<float>(z + depth);
            const float y = static_cast<float>(faceY);
            addFace(
                subchunkLocalY,
                {{
                    {fx, y, fz},
                    {fx, y, zEnd},
                    {xEnd, y, zEnd},
                    {xEnd, y, fz},
                }},
                {{
                    {{0.0f, 0.0f}},
                    {{static_cast<float>(depth), 0.0f}},
                    {{static_cast<float>(depth), static_cast<float>(width)}},
                    {{0.0f, static_cast<float>(width)}},
                }},
                textureLayer,
                ao);
        };

        auto addBottomFace = [&](
            int subchunkLocalY,
            int faceY,
            int x,
            int z,
            int width,
            int depth,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            const float fx = static_cast<float>(x);
            const float fz = static_cast<float>(z);
            const float xEnd = static_cast<float>(x + width);
            const float zEnd = static_cast<float>(z + depth);
            const float y = static_cast<float>(faceY);
            addFace(
                subchunkLocalY,
                {{
                    {fx, y, zEnd},
                    {fx, y, fz},
                    {xEnd, y, fz},
                    {xEnd, y, zEnd},
                }},
                {{
                    {{0.0f, 0.0f}},
                    {{static_cast<float>(depth), 0.0f}},
                    {{static_cast<float>(depth), static_cast<float>(width)}},
                    {{0.0f, static_cast<float>(width)}},
                }},
                textureLayer,
                ao);
        };

        auto addSideFace = [&](
            int subchunkLocalY,
            std::array<Vec3, 4> corners,
            int width,
            int height,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            addFace(
                subchunkLocalY,
                corners,
                {{
                    {{static_cast<float>(width), static_cast<float>(height)}},
                    {{static_cast<float>(width), 0.0f}},
                    {{0.0f, 0.0f}},
                    {{0.0f, static_cast<float>(height)}},
                }},
                textureLayer,
                ao);
        };

        for (auto& vertices : verticesByLocalY)
        {
            vertices.clear();
        }
        for (auto& indices : indicesByLocalY)
        {
            indices.clear();
        }

            for (int localY = 0; localY < kSubchunkSize; ++localY)
            {
                const int faceY = subchunkMinY + localY + 1;
                emitGreedyRectangles(
                    kChunkSizeX,
                    kChunkSizeZ,
                    [&](int localX, int localZ) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int z = chunkBaseZ + localZ;
                        if (!(isSolid(x, faceY - 1, z) && !isSolid(x, faceY, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, faceY - 1, z), BlockFace::Top),
                            topFaceAo(x, z, faceY),
                        };
                    },
                    [&](int localX, int localZ, int width, int depth, const FaceMaskCell& cell)
                    {
                        addTopFace(
                            localY,
                            faceY,
                            chunkBaseX + localX,
                            chunkBaseZ + localZ,
                            width,
                            depth,
                            cell.textureLayer,
                            cell.ao);
                    });

                const int bottomFaceY = subchunkMinY + localY;
                emitGreedyRectangles(
                    kChunkSizeX,
                    kChunkSizeZ,
                    [&](int localX, int localZ) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int z = chunkBaseZ + localZ;
                        if (!(isSolid(x, bottomFaceY, z) && !isSolid(x, bottomFaceY - 1, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, bottomFaceY, z), BlockFace::Bottom),
                            bottomFaceAo(x, z, bottomFaceY),
                        };
                    },
                    [&](int localX, int localZ, int width, int depth, const FaceMaskCell& cell)
                    {
                        addBottomFace(
                            localY,
                            bottomFaceY,
                            chunkBaseX + localX,
                            chunkBaseZ + localZ,
                            width,
                            depth,
                            cell.textureLayer,
                            cell.ao);
                    });
            }

            for (int localX = 0; localX < kChunkSizeX; ++localX)
            {
                const int x = chunkBaseX + localX;
                const float faceX = static_cast<float>(x + 1);
                emitGreedyRectangles(
                    kChunkSizeZ,
                    kSubchunkSize,
                    [&](int localZ, int localY) -> FaceMaskCell
                    {
                        const int z = chunkBaseZ + localZ;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x + 1, y, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            positiveXFaceAo(x, y, z),
                        };
                    },
                    [&](int localZ, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float z0 = static_cast<float>(chunkBaseZ + localZ);
                        const float z1 = static_cast<float>(chunkBaseZ + localZ + width);
                        addSideFace(
                            localY,
                            {{
                            {faceX, bottom, z0},
                            {faceX, top, z0},
                            {faceX, top, z1},
                            {faceX, bottom, z1},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });

                const float oppositeFaceX = static_cast<float>(x);
                emitGreedyRectangles(
                    kChunkSizeZ,
                    kSubchunkSize,
                    [&](int localZ, int localY) -> FaceMaskCell
                    {
                        const int z = chunkBaseZ + localZ;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x - 1, y, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            negativeXFaceAo(x, y, z),
                        };
                    },
                    [&](int localZ, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float z0 = static_cast<float>(chunkBaseZ + localZ);
                        const float z1 = static_cast<float>(chunkBaseZ + localZ + width);
                        addSideFace(
                            localY,
                            {{
                                {oppositeFaceX, bottom, z1},
                                {oppositeFaceX, top, z1},
                            {oppositeFaceX, top, z0},
                            {oppositeFaceX, bottom, z0},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });
            }

            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
            {
                const int z = chunkBaseZ + localZ;
                const float faceZ = static_cast<float>(z + 1);
                emitGreedyRectangles(
                    kChunkSizeX,
                    kSubchunkSize,
                    [&](int localX, int localY) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x, y, z + 1)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            positiveZFaceAo(x, y, z),
                        };
                    },
                    [&](int localX, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float x0 = static_cast<float>(chunkBaseX + localX);
                        const float x1 = static_cast<float>(chunkBaseX + localX + width);
                        addSideFace(
                            localY,
                            {{
                                {x1, bottom, faceZ},
                                {x1, top, faceZ},
                            {x0, top, faceZ},
                            {x0, bottom, faceZ},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });

                const float oppositeFaceZ = static_cast<float>(z);
                emitGreedyRectangles(
                    kChunkSizeX,
                    kSubchunkSize,
                    [&](int localX, int localY) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x, y, z - 1)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            negativeZFaceAo(x, y, z),
                        };
                    },
                    [&](int localX, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float x0 = static_cast<float>(chunkBaseX + localX);
                        const float x1 = static_cast<float>(chunkBaseX + localX + width);
                        addSideFace(
                            localY,
                            {{
                                {x0, bottom, oppositeFaceZ},
                                {x0, top, oppositeFaceZ},
                            {x1, top, oppositeFaceZ},
                            {x1, bottom, oppositeFaceZ},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });
            }

            auto fluidTop = [&](int x, int y, int z, std::uint8_t fluidId, std::uint8_t amount) -> float
            {
                if (amount == kMaxFluidAmount &&
                    fluidIdAt(x, y + 1, z) == fluidId &&
                    fluidAmountAt(x, y + 1, z) == kMaxFluidAmount)
                {
                    return static_cast<float>(y + 1);
                }
                return static_cast<float>(y) + 0.9f *
                    (static_cast<float>(amount) / static_cast<float>(kMaxFluidAmount));
            };

            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
            {
                const int z = chunkBaseZ + localZ;
                for (int localX = 0; localX < kChunkSizeX; ++localX)
                {
                    const int x = chunkBaseX + localX;
                    for (int localY = 0; localY < kSubchunkSize; ++localY)
                    {
                        const int y = subchunkMinY + localY;
                        const std::uint8_t fluidId = fluidIdAt(x, y, z);
                        const std::uint8_t amount = fluidAmountAt(x, y, z);
                        if (fluidId != kWaterFluidId || amount == 0 || isSolid(x, y, z))
                        {
                            continue;
                        }

                        const float bottom = static_cast<float>(y);
                        const float top = fluidTop(x, y, z, fluidId, amount);
                        const float fx = static_cast<float>(x);
                        const float fz = static_cast<float>(z);
                        const float xEnd = static_cast<float>(x + 1);
                        const float zEnd = static_cast<float>(z + 1);

                        if ((fluidIdAt(x, y + 1, z) != fluidId || fluidAmountAt(x, y + 1, z) == 0) &&
                            !isSolid(x, y + 1, z))
                        {
                            addFluidFace(
                                localY,
                                {{
                                    {fx, top, fz},
                                    {fx, top, zEnd},
                                    {xEnd, top, zEnd},
                                    {xEnd, top, fz},
                                }},
                                1.0f,
                                1.0f);
                        }

                        if ((fluidIdAt(x, y - 1, z) != fluidId || fluidAmountAt(x, y - 1, z) == 0) &&
                            !isSolid(x, y - 1, z))
                        {
                            addFluidFace(
                                localY,
                                {{
                                    {fx, bottom, zEnd},
                                    {fx, bottom, fz},
                                    {xEnd, bottom, fz},
                                    {xEnd, bottom, zEnd},
                                }},
                                1.0f,
                                1.0f);
                        }

                        auto addFluidSideIfVisible = [&](
                            int neighborX,
                            int neighborZ,
                            std::array<Vec3, 4> fullCorners)
                        {
                            if (isSolid(neighborX, y, neighborZ))
                            {
                                return;
                            }

                            float sideBottom = bottom;
                            const std::uint8_t neighborFluidId = fluidIdAt(neighborX, y, neighborZ);
                            const std::uint8_t neighborAmount = fluidAmountAt(neighborX, y, neighborZ);
                            if (neighborFluidId == fluidId && neighborAmount > 0)
                            {
                                sideBottom = fluidTop(neighborX, y, neighborZ, neighborFluidId, neighborAmount);
                                if (sideBottom >= top)
                                {
                                    return;
                                }
                            }

                            fullCorners[0].y = sideBottom;
                            fullCorners[1].y = top;
                            fullCorners[2].y = top;
                            fullCorners[3].y = sideBottom;
                            addFluidFace(localY, fullCorners, 1.0f, top - sideBottom);
                        };

                        addFluidSideIfVisible(
                            x + 1,
                            z,
                            {{
                                {xEnd, bottom, fz},
                                {xEnd, top, fz},
                                {xEnd, top, zEnd},
                                {xEnd, bottom, zEnd},
                            }});
                        addFluidSideIfVisible(
                            x - 1,
                            z,
                            {{
                                {fx, bottom, zEnd},
                                {fx, top, zEnd},
                                {fx, top, fz},
                                {fx, bottom, fz},
                            }});
                        addFluidSideIfVisible(
                            x,
                            z + 1,
                            {{
                                {xEnd, bottom, zEnd},
                                {xEnd, top, zEnd},
                                {fx, top, zEnd},
                                {fx, bottom, zEnd},
                            }});
                        addFluidSideIfVisible(
                            x,
                            z - 1,
                            {{
                                {fx, bottom, fz},
                                {fx, top, fz},
                                {xEnd, top, fz},
                                {xEnd, bottom, fz},
                            }});
                    }
                }
            }

        appendSubchunkMesh(
            chunkX,
            chunkZ,
            subchunkY,
            verticesByLocalY,
            indicesByLocalY,
            subchunkVertices,
            subchunkIndices,
            subchunkDraws);

        return {
            chunk,
            subchunkY,
            request.generation,
            std::move(subchunkVertices),
            std::move(subchunkIndices),
        };
    }

void ChunkMesher::appendSubchunkMesh(
        int chunkX,
        int chunkZ,
        int subchunkY,
        const std::array<std::vector<BlockVertex>, kSubchunkSize>& verticesByLocalY,
        const std::array<std::vector<std::uint32_t>, kSubchunkSize>& indicesByLocalY,
        std::vector<BlockVertex>& chunkVertices,
        std::vector<std::uint32_t>& chunkIndices,
        std::vector<SubchunkDraw>& subchunkDraws) const
{
        std::size_t vertexCount = 0;
        std::size_t indexCount = 0;
        for (int localY = 0; localY < kSubchunkSize; ++localY)
        {
            vertexCount += verticesByLocalY[static_cast<std::size_t>(localY)].size();
            indexCount += indicesByLocalY[static_cast<std::size_t>(localY)].size();
        }

        if (indexCount == 0)
        {
            return;
        }

        const std::uint32_t vertexOffset = static_cast<std::uint32_t>(chunkVertices.size());
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(chunkIndices.size());
        std::uint32_t subchunkVertexCount = 0;

        for (int localY = 0; localY < kSubchunkSize; ++localY)
        {
            const auto& sourceVertices = verticesByLocalY[static_cast<std::size_t>(localY)];
            const auto& sourceIndices = indicesByLocalY[static_cast<std::size_t>(localY)];
            const std::uint32_t baseVertex = subchunkVertexCount;

            chunkVertices.insert(chunkVertices.end(), sourceVertices.begin(), sourceVertices.end());
            for (std::uint32_t index : sourceIndices)
            {
                chunkIndices.push_back(baseVertex + index);
            }
            subchunkVertexCount += static_cast<std::uint32_t>(sourceVertices.size());
        }

        SubchunkDraw draw{};
        draw.chunkX = chunkX;
        draw.chunkZ = chunkZ;
        draw.subchunkY = subchunkY;
        draw.range.vertexCount = static_cast<std::uint32_t>(vertexCount);
        draw.range.firstIndex = firstIndex;
        draw.range.indexCount = static_cast<std::uint32_t>(indexCount);
        draw.range.vertexOffset = static_cast<std::int32_t>(vertexOffset);
        subchunkDraws.push_back(draw);
    }


