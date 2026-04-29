#pragma once

#include "player/PlayerTypes.h"
#include "world/ChunkTypes.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class Logger;

struct WorldSaveState
{
    CameraState camera{};
    MovementMode movementMode = MovementMode::Fly;
    CameraViewMode cameraViewMode = CameraViewMode::FirstPerson;
    std::uint64_t worldTimeTicks = 0;
    std::uint64_t worldSeed = 0;
};

class WorldSave
{
public:
    void initialize(const std::wstring& worldDirectory, Logger* logger);

    std::optional<WorldSaveState> loadWorldState();
    void saveWorldState(const WorldSaveState& state);

    std::optional<ChunkVoxelData> loadChunk(ChunkCoord coord);
    void saveChunk(ChunkCoord coord, const ChunkVoxelData& voxels);

private:
    struct ChunkEntry
    {
        std::uint32_t offsetSector = 0;
        std::uint32_t sectorCount = 0;
        std::uint32_t byteSize = 0;
        std::uint32_t reserved = 0;
    };

    static constexpr int kRegionChunkSide = 16;
    static constexpr int kRegionChunkCount = kRegionChunkSide * kRegionChunkSide;
    static constexpr std::uint32_t kSectorSize = 4096;
    static constexpr std::uint32_t kHeaderSectors = 1;
    static constexpr std::uint32_t kHeaderSize = kSectorSize * kHeaderSectors;

    static int floorDiv(int value, int divisor);
    static int floorMod(int value, int divisor);
    static int regionCoordForChunk(int chunkCoord);
    static int localCoordForChunk(int chunkCoord);
    static int chunkTableIndex(int localChunkX, int localChunkZ);
    static std::uint32_t sectorsForBytes(std::uint32_t byteSize);

    std::wstring regionPath(int regionX, int regionZ) const;
    std::vector<ChunkEntry> readChunkTable(const std::wstring& path);
    void writeChunkTable(const std::wstring& path, const std::vector<ChunkEntry>& entries);
    void ensureRegionFile(const std::wstring& path);
    std::uint32_t appendOffsetSector(const std::wstring& path) const;

    static void appendU16(std::vector<std::uint8_t>& bytes, std::uint16_t value);
    static void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value);
    static std::uint16_t readU16(const std::vector<std::uint8_t>& bytes, std::size_t& offset);
    static std::uint32_t readU32(const std::vector<std::uint8_t>& bytes, std::size_t& offset);
    static std::vector<std::uint8_t> encodeRle(const ChunkVoxelData& voxels);
    static ChunkVoxelData decodeRle(const std::vector<std::uint8_t>& bytes);
    static std::vector<std::uint8_t> compressLz4(const std::vector<std::uint8_t>& bytes);
    static std::vector<std::uint8_t> decompressLz4(const std::vector<std::uint8_t>& bytes);

    std::wstring worldDirectory_;
    std::wstring regionsDirectory_;
    Logger* logger_ = nullptr;
};
