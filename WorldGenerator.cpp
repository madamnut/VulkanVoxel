#include "BlockRegistry.h"
#include "PeriodicSimplex.h"
#include "WorldGenerator.h"

#include <algorithm>
#include <vector>

namespace WorldGenerator {

namespace {

constexpr int kDensitySampleStepX = 4;
constexpr int kDensitySampleStepY = 4;
constexpr int kDensitySampleStepZ = 4;
constexpr int kDensityGridCountX = kChunkSizeX / kDensitySampleStepX + 1;
constexpr int kDensityGridCountY = kWorldSizeY / kDensitySampleStepY + 1;
constexpr int kDensityGridCountZ = kChunkSizeZ / kDensitySampleStepZ + 1;

static_assert(kChunkSizeX % kDensitySampleStepX == 0);
static_assert(kWorldSizeY % kDensitySampleStepY == 0);
static_assert(kChunkSizeZ % kDensitySampleStepZ == 0);

PeriodicSimplex::FbmSettings MakeNoiseSettings(const TerrainConfig& terrainConfig) {
    PeriodicSimplex::FbmSettings settings{};
    settings.seed = terrainConfig.seed;
    settings.wrapSizeXZ = terrainConfig.wrapSizeXZ;
    settings.featureScaleXZ = terrainConfig.featureScaleXZ;
    settings.featureScaleY = terrainConfig.featureScaleY;
    settings.octaves = terrainConfig.fbmOctaves;
    settings.gain = terrainConfig.fbmGain;
    settings.lacunarity = terrainConfig.fbmLacunarity;
    return settings;
}

float Lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

int FloorToSampleStep(int value, int step) {
    const int remainder = value % step;
    if (remainder < 0) {
        return value - (remainder + step);
    }

    return value - remainder;
}

float GetAxisInterpolationFactor(int value, int step, int lower) {
    return static_cast<float>(value - lower) / static_cast<float>(step);
}

float SampleRawTerrainDensity(
    int worldX,
    int worldY,
    int worldZ,
    const TerrainConfig& terrainConfig,
    const PeriodicSimplex::FbmSettings& noiseSettings
) {
    const float periodicNoise = PeriodicSimplex::SampleTileableXZ3DFbm(
        static_cast<float>(worldX),
        static_cast<float>(worldY),
        static_cast<float>(worldZ),
        noiseSettings
    );
    const float verticalGradient =
        (terrainConfig.baseHeight - static_cast<float>(worldY)) * terrainConfig.gradientStrength;
    return verticalGradient + periodicNoise * terrainConfig.densityAmplitude;
}

struct ChunkDensityGrid {
    std::vector<float> samples;

    float& At(int sampleX, int sampleY, int sampleZ) {
        const std::size_t index = static_cast<std::size_t>(
            (sampleY * kDensityGridCountZ + sampleZ) * kDensityGridCountX + sampleX);
        return samples[index];
    }

    float At(int sampleX, int sampleY, int sampleZ) const {
        const std::size_t index = static_cast<std::size_t>(
            (sampleY * kDensityGridCountZ + sampleZ) * kDensityGridCountX + sampleX);
        return samples[index];
    }
};

ChunkDensityGrid BuildChunkDensityGrid(
    int chunkX,
    int chunkZ,
    const TerrainConfig& terrainConfig,
    const PeriodicSimplex::FbmSettings& noiseSettings
) {
    ChunkDensityGrid grid{};
    grid.samples.resize(static_cast<std::size_t>(kDensityGridCountX * kDensityGridCountY * kDensityGridCountZ));

    const int chunkMinWorldX = chunkX * kChunkSizeX;
    const int chunkMinWorldZ = chunkZ * kChunkSizeZ;

    for (int sampleY = 0; sampleY < kDensityGridCountY; ++sampleY) {
        const int worldY = sampleY * kDensitySampleStepY;
        for (int sampleZ = 0; sampleZ < kDensityGridCountZ; ++sampleZ) {
            const int worldZ = chunkMinWorldZ + sampleZ * kDensitySampleStepZ;
            for (int sampleX = 0; sampleX < kDensityGridCountX; ++sampleX) {
                const int worldX = chunkMinWorldX + sampleX * kDensitySampleStepX;
                grid.At(sampleX, sampleY, sampleZ) =
                    SampleRawTerrainDensity(worldX, worldY, worldZ, terrainConfig, noiseSettings);
            }
        }
    }

    return grid;
}

float SampleInterpolatedTerrainDensityFromChunkGrid(
    const ChunkDensityGrid& densityGrid,
    int localX,
    int worldY,
    int localZ
) {
    const int sampleX0 = localX / kDensitySampleStepX;
    const int sampleY0 = worldY / kDensitySampleStepY;
    const int sampleZ0 = localZ / kDensitySampleStepZ;
    const int sampleX1 = sampleX0 + 1;
    const int sampleY1 = sampleY0 + 1;
    const int sampleZ1 = sampleZ0 + 1;

    const float tx = static_cast<float>(localX % kDensitySampleStepX) / static_cast<float>(kDensitySampleStepX);
    const float ty = static_cast<float>(worldY % kDensitySampleStepY) / static_cast<float>(kDensitySampleStepY);
    const float tz = static_cast<float>(localZ % kDensitySampleStepZ) / static_cast<float>(kDensitySampleStepZ);

    const float c000 = densityGrid.At(sampleX0, sampleY0, sampleZ0);
    const float c100 = densityGrid.At(sampleX1, sampleY0, sampleZ0);
    const float c010 = densityGrid.At(sampleX0, sampleY1, sampleZ0);
    const float c110 = densityGrid.At(sampleX1, sampleY1, sampleZ0);
    const float c001 = densityGrid.At(sampleX0, sampleY0, sampleZ1);
    const float c101 = densityGrid.At(sampleX1, sampleY0, sampleZ1);
    const float c011 = densityGrid.At(sampleX0, sampleY1, sampleZ1);
    const float c111 = densityGrid.At(sampleX1, sampleY1, sampleZ1);

    const float c00 = Lerp(c000, c100, tx);
    const float c10 = Lerp(c010, c110, tx);
    const float c01 = Lerp(c001, c101, tx);
    const float c11 = Lerp(c011, c111, tx);
    const float c0 = Lerp(c00, c10, ty);
    const float c1 = Lerp(c01, c11, ty);
    return Lerp(c0, c1, tz);
}

float SampleInterpolatedTerrainDensity(
    int worldX,
    int worldY,
    int worldZ,
    const TerrainConfig& terrainConfig,
    const PeriodicSimplex::FbmSettings& noiseSettings
) {
    const int x0 = FloorToSampleStep(worldX, kDensitySampleStepX);
    const int y0 = FloorToSampleStep(worldY, kDensitySampleStepY);
    const int z0 = FloorToSampleStep(worldZ, kDensitySampleStepZ);
    const int x1 = x0 + kDensitySampleStepX;
    const int y1 = y0 + kDensitySampleStepY;
    const int z1 = z0 + kDensitySampleStepZ;

    const float tx = GetAxisInterpolationFactor(worldX, kDensitySampleStepX, x0);
    const float ty = GetAxisInterpolationFactor(worldY, kDensitySampleStepY, y0);
    const float tz = GetAxisInterpolationFactor(worldZ, kDensitySampleStepZ, z0);

    const float c000 = SampleRawTerrainDensity(x0, y0, z0, terrainConfig, noiseSettings);
    const float c100 = SampleRawTerrainDensity(x1, y0, z0, terrainConfig, noiseSettings);
    const float c010 = SampleRawTerrainDensity(x0, y1, z0, terrainConfig, noiseSettings);
    const float c110 = SampleRawTerrainDensity(x1, y1, z0, terrainConfig, noiseSettings);
    const float c001 = SampleRawTerrainDensity(x0, y0, z1, terrainConfig, noiseSettings);
    const float c101 = SampleRawTerrainDensity(x1, y0, z1, terrainConfig, noiseSettings);
    const float c011 = SampleRawTerrainDensity(x0, y1, z1, terrainConfig, noiseSettings);
    const float c111 = SampleRawTerrainDensity(x1, y1, z1, terrainConfig, noiseSettings);

    const float c00 = Lerp(c000, c100, tx);
    const float c10 = Lerp(c010, c110, tx);
    const float c01 = Lerp(c001, c101, tx);
    const float c11 = Lerp(c011, c111, tx);
    const float c0 = Lerp(c00, c10, ty);
    const float c1 = Lerp(c01, c11, ty);
    return Lerp(c0, c1, tz);
}

void TryCollapseGeneratedSubChunk(SubChunkVoxelData& subChunk) {
    if (subChunk.isUniform || subChunk.blocks.empty()) {
        return;
    }

    const std::uint16_t candidate = subChunk.blocks.front();
    for (std::uint16_t blockValue : subChunk.blocks) {
        if (blockValue != candidate) {
            return;
        }
    }

    subChunk.isUniform = true;
    subChunk.uniformBlock = candidate;
    subChunk.blocks.clear();
    subChunk.blocks.shrink_to_fit();
}

}  // namespace

int ComputeTerrainHeight(int worldX, int worldZ, const TerrainConfig& terrainConfig) {
    const PeriodicSimplex::FbmSettings noiseSettings = MakeNoiseSettings(terrainConfig);
    for (int worldY = kWorldSizeY - 1; worldY >= 0; --worldY) {
        if (SampleInterpolatedTerrainDensity(worldX, worldY, worldZ, terrainConfig, noiseSettings) >
            terrainConfig.solidThreshold) {
            return worldY + 1;
        }
    }

    return 0;
}

std::uint16_t SampleBlock(int worldX, int worldY, int worldZ, const TerrainConfig& terrainConfig) {
    if (worldY < 0 || worldY >= kWorldSizeY) {
        return kBlockAir;
    }

    const float density =
        SampleInterpolatedTerrainDensity(worldX, worldY, worldZ, terrainConfig, MakeNoiseSettings(terrainConfig));
    return density > terrainConfig.solidThreshold ? kBlockRock : kBlockAir;
}

void GenerateChunkColumn(int chunkX, int chunkZ, const TerrainConfig& terrainConfig, ChunkColumnData& outColumn) {
    outColumn.subChunks.clear();
    outColumn.subChunks.resize(kSubChunkCountY);
    outColumn.subChunkMeshes.clear();
    outColumn.subChunkMeshes.resize(kSubChunkCountY);
    outColumn.modified = false;

    for (SubChunkVoxelData& subChunk : outColumn.subChunks) {
        subChunk.blocks.clear();
        subChunk.uniformBlock = 0;
        subChunk.isUniform = true;
    }

    for (SubChunkMeshData& subChunkMesh : outColumn.subChunkMeshes) {
        subChunkMesh.quads.clear();
        subChunkMesh.dirty = true;
        subChunkMesh.revision = 0;
    }

    const PeriodicSimplex::FbmSettings noiseSettings = MakeNoiseSettings(terrainConfig);
    const ChunkDensityGrid densityGrid = BuildChunkDensityGrid(chunkX, chunkZ, terrainConfig, noiseSettings);

    for (int subChunkIndex = 0; subChunkIndex < kSubChunkCountY; ++subChunkIndex) {
        const int subChunkMinY = subChunkIndex * kSubChunkSize;
        SubChunkVoxelData& subChunk = outColumn.subChunks[static_cast<std::size_t>(subChunkIndex)];
        subChunk.isUniform = false;
        subChunk.uniformBlock = kBlockAir;
        subChunk.blocks.assign(static_cast<std::size_t>(kSubChunkVoxelCount), kBlockAir);

        bool isUniform = true;
        std::uint16_t firstBlock = kBlockAir;
        bool firstBlockInitialized = false;

        for (int localZ = 0; localZ < kChunkSizeZ; ++localZ) {
            for (int localX = 0; localX < kChunkSizeX; ++localX) {
                for (int localY = 0; localY < kSubChunkSize; ++localY) {
                    const int worldY = subChunkMinY + localY;
                    const int blockIndex = localY * kSubChunkSize * kSubChunkSize +
                        localZ * kSubChunkSize + localX;
                    const float density =
                        SampleInterpolatedTerrainDensityFromChunkGrid(densityGrid, localX, worldY, localZ);
                    const std::uint16_t blockValue =
                        density > terrainConfig.solidThreshold ? kBlockRock : kBlockAir;
                    subChunk.blocks[static_cast<std::size_t>(blockIndex)] = blockValue;

                    if (!firstBlockInitialized) {
                        firstBlock = blockValue;
                        firstBlockInitialized = true;
                    } else if (blockValue != firstBlock) {
                        isUniform = false;
                    }
                }
            }
        }

        if (isUniform) {
            subChunk.isUniform = true;
            subChunk.uniformBlock = firstBlock;
            subChunk.blocks.clear();
            continue;
        }

        TryCollapseGeneratedSubChunk(subChunk);
    }
}

}  // namespace WorldGenerator
