#include "world/WorldSave.h"

#include "core/Logger.h"
#include "world/Block.h"
#include "lz4.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace
{
constexpr std::size_t kMaxRleBytes = 4 + kChunkBlockCount * 8;

template <typename T>
void writePod(std::ostream& stream, const T& value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool readPod(std::istream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

std::string narrowPath(const std::wstring& value)
{
    return std::string(value.begin(), value.end());
}
}

void WorldSave::initialize(const std::wstring& worldDirectory, Logger* logger)
{
    worldDirectory_ = worldDirectory;
    regionsDirectory_ = (std::filesystem::path(worldDirectory_) / L"regions").wstring();
    logger_ = logger;

    std::filesystem::create_directories(std::filesystem::path(regionsDirectory_));
    if (logger_ != nullptr)
    {
        logger_->info("Save directory: " + narrowPath(worldDirectory_));
    }
}

std::optional<WorldSaveState> WorldSave::loadWorldState()
{
    const std::filesystem::path path = std::filesystem::path(worldDirectory_) / L"world.bin";
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        if (logger_ != nullptr)
        {
            logger_->warn("Missing world.bin, starting new world state");
        }
        return std::nullopt;
    }

    WorldSaveState state{};
    std::uint8_t movementMode = 0;
    std::uint8_t cameraViewMode = 0;
    if (!readPod(stream, state.camera.position.x) ||
        !readPod(stream, state.camera.position.y) ||
        !readPod(stream, state.camera.position.z) ||
        !readPod(stream, state.camera.yaw) ||
        !readPod(stream, state.camera.pitch) ||
        !readPod(stream, movementMode) ||
        !readPod(stream, cameraViewMode) ||
        !readPod(stream, state.worldTimeTicks) ||
        !readPod(stream, state.worldSeed))
    {
        if (logger_ != nullptr)
        {
            logger_->warn("Failed to read world.bin, starting new world state");
        }
        return std::nullopt;
    }

    state.movementMode = movementMode == 1 ? MovementMode::Ground : MovementMode::Fly;
    if (cameraViewMode == 1)
    {
        state.cameraViewMode = CameraViewMode::ThirdPersonRear;
    }
    else if (cameraViewMode == 2)
    {
        state.cameraViewMode = CameraViewMode::ThirdPersonFront;
    }
    else
    {
        state.cameraViewMode = CameraViewMode::FirstPerson;
    }

    if (logger_ != nullptr)
    {
        logger_->info("Loaded world.bin");
    }
    return state;
}

void WorldSave::saveWorldState(const WorldSaveState& state)
{
    std::filesystem::create_directories(std::filesystem::path(worldDirectory_));
    const std::filesystem::path path = std::filesystem::path(worldDirectory_) / L"world.bin";
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("Failed to write world.bin.");
    }

    const std::uint8_t movementMode = state.movementMode == MovementMode::Ground ? 1 : 0;
    std::uint8_t cameraViewMode = 0;
    if (state.cameraViewMode == CameraViewMode::ThirdPersonRear)
    {
        cameraViewMode = 1;
    }
    else if (state.cameraViewMode == CameraViewMode::ThirdPersonFront)
    {
        cameraViewMode = 2;
    }

    writePod(stream, state.camera.position.x);
    writePod(stream, state.camera.position.y);
    writePod(stream, state.camera.position.z);
    writePod(stream, state.camera.yaw);
    writePod(stream, state.camera.pitch);
    writePod(stream, movementMode);
    writePod(stream, cameraViewMode);
    writePod(stream, state.worldTimeTicks);
    writePod(stream, state.worldSeed);

    if (logger_ != nullptr)
    {
        logger_->info("Saved world.bin");
    }
}

std::optional<ChunkVoxelData> WorldSave::loadChunk(ChunkCoord coord)
{
    const int wrappedChunkX = floorMod(coord.x, kWorldChunkSide);
    const int wrappedChunkZ = floorMod(coord.z, kWorldChunkSide);
    const int regionX = regionCoordForChunk(wrappedChunkX);
    const int regionZ = regionCoordForChunk(wrappedChunkZ);
    const int localChunkX = localCoordForChunk(wrappedChunkX);
    const int localChunkZ = localCoordForChunk(wrappedChunkZ);
    const std::wstring path = regionPath(regionX, regionZ);
    if (!std::filesystem::exists(std::filesystem::path(path)))
    {
        return std::nullopt;
    }

    std::vector<ChunkEntry> entries;
    try
    {
        entries = readChunkTable(path);
    }
    catch (const std::exception& exception)
    {
        if (logger_ != nullptr)
        {
            logger_->warn(std::string("Failed to read region table; generating chunk: ") + exception.what());
        }
        return std::nullopt;
    }

    const ChunkEntry& entry = entries[static_cast<std::size_t>(chunkTableIndex(localChunkX, localChunkZ))];
    if (entry.offsetSector == 0 || entry.byteSize == 0)
    {
        return std::nullopt;
    }
    if (entry.sectorCount == 0 ||
        entry.byteSize > entry.sectorCount * kSectorSize)
    {
        if (logger_ != nullptr)
        {
            logger_->warn("Invalid chunk entry size; generating chunk");
        }
        return std::nullopt;
    }

    std::ifstream stream(std::filesystem::path(path), std::ios::binary);
    if (!stream)
    {
        if (logger_ != nullptr)
        {
            logger_->warn("Failed to open region for chunk load: " + narrowPath(path));
        }
        return std::nullopt;
    }

    std::vector<std::uint8_t> payload(entry.byteSize);
    stream.seekg(static_cast<std::streamoff>(entry.offsetSector) * kSectorSize, std::ios::beg);
    stream.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!stream)
    {
        if (logger_ != nullptr)
        {
            logger_->warn("Failed to read chunk payload; generating chunk");
        }
        return std::nullopt;
    }

    try
    {
        return decodeRle(decompressLz4(payload));
    }
    catch (const std::exception& exception)
    {
        if (logger_ != nullptr)
        {
            logger_->warn(std::string("Failed to decode chunk payload; generating chunk: ") + exception.what());
        }
        return std::nullopt;
    }
}

void WorldSave::saveChunk(ChunkCoord coord, const ChunkVoxelData& voxels)
{
    if (!voxels.valid())
    {
        throw std::runtime_error("Cannot save chunk with invalid voxel data.");
    }

    const int wrappedChunkX = floorMod(coord.x, kWorldChunkSide);
    const int wrappedChunkZ = floorMod(coord.z, kWorldChunkSide);
    const int regionX = regionCoordForChunk(wrappedChunkX);
    const int regionZ = regionCoordForChunk(wrappedChunkZ);
    const int localChunkX = localCoordForChunk(wrappedChunkX);
    const int localChunkZ = localCoordForChunk(wrappedChunkZ);
    const std::wstring path = regionPath(regionX, regionZ);
    ensureRegionFile(path);

    std::vector<ChunkEntry> entries = readChunkTable(path);
    ChunkEntry& entry = entries[static_cast<std::size_t>(chunkTableIndex(localChunkX, localChunkZ))];

    const std::vector<std::uint8_t> payload = compressLz4(encodeRle(voxels));
    const std::uint32_t byteSize = static_cast<std::uint32_t>(payload.size());
    const std::uint32_t requiredSectors = sectorsForBytes(byteSize);
    std::uint32_t writeSector = entry.offsetSector;
    const std::uint32_t previousSector = entry.offsetSector;
    const std::uint32_t previousSectorCount = entry.sectorCount;
    if (writeSector == 0 || requiredSectors > entry.sectorCount)
    {
        writeSector = appendOffsetSector(path);
    }

    if (previousSector != 0 && previousSectorCount > 0)
    {
        std::fstream clearStream(std::filesystem::path(path), std::ios::binary | std::ios::in | std::ios::out);
        std::vector<char> zeroes(static_cast<std::size_t>(previousSectorCount) * kSectorSize, 0);
        clearStream.seekp(static_cast<std::streamoff>(previousSector) * kSectorSize, std::ios::beg);
        clearStream.write(zeroes.data(), static_cast<std::streamsize>(zeroes.size()));
    }

    std::fstream stream(std::filesystem::path(path), std::ios::binary | std::ios::in | std::ios::out);
    if (!stream)
    {
        throw std::runtime_error("Failed to open region for chunk save.");
    }

    stream.seekp(static_cast<std::streamoff>(writeSector) * kSectorSize, std::ios::beg);
    stream.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    const std::size_t paddedSize = static_cast<std::size_t>(requiredSectors) * kSectorSize;
    if (paddedSize > payload.size())
    {
        std::vector<char> zeroes(paddedSize - payload.size(), 0);
        stream.write(zeroes.data(), static_cast<std::streamsize>(zeroes.size()));
    }
    if (!stream)
    {
        throw std::runtime_error("Failed to write chunk payload.");
    }

    entry.offsetSector = writeSector;
    entry.sectorCount = requiredSectors;
    entry.byteSize = byteSize;
    writeChunkTable(path, entries);
}

int WorldSave::floorDiv(int value, int divisor)
{
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

int WorldSave::floorMod(int value, int divisor)
{
    int result = value % divisor;
    if (result < 0)
    {
        result += divisor;
    }
    return result;
}

int WorldSave::regionCoordForChunk(int chunkCoord)
{
    return floorDiv(chunkCoord, kRegionChunkSide);
}

int WorldSave::localCoordForChunk(int chunkCoord)
{
    return floorMod(chunkCoord, kRegionChunkSide);
}

int WorldSave::chunkTableIndex(int localChunkX, int localChunkZ)
{
    return localChunkZ * kRegionChunkSide + localChunkX;
}

std::uint32_t WorldSave::sectorsForBytes(std::uint32_t byteSize)
{
    return std::max<std::uint32_t>(1, (byteSize + kSectorSize - 1) / kSectorSize);
}

std::wstring WorldSave::regionPath(int regionX, int regionZ) const
{
    return regionsDirectory_ +
        L"\\r." + std::to_wstring(regionX) +
        L"." + std::to_wstring(regionZ) +
        L".vvr";
}

std::vector<WorldSave::ChunkEntry> WorldSave::readChunkTable(const std::wstring& path)
{
    ensureRegionFile(path);
    std::ifstream stream(std::filesystem::path(path), std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Failed to read region chunk table.");
    }

    std::vector<ChunkEntry> entries(kRegionChunkCount);
    for (ChunkEntry& entry : entries)
    {
        if (!readPod(stream, entry.offsetSector) ||
            !readPod(stream, entry.sectorCount) ||
            !readPod(stream, entry.byteSize) ||
            !readPod(stream, entry.reserved))
        {
            throw std::runtime_error("Region chunk table is truncated.");
        }
    }
    return entries;
}

void WorldSave::writeChunkTable(const std::wstring& path, const std::vector<ChunkEntry>& entries)
{
    if (entries.size() != kRegionChunkCount)
    {
        throw std::runtime_error("Invalid region chunk table size.");
    }

    std::fstream stream(std::filesystem::path(path), std::ios::binary | std::ios::in | std::ios::out);
    if (!stream)
    {
        throw std::runtime_error("Failed to write region chunk table.");
    }

    stream.seekp(0, std::ios::beg);
    for (const ChunkEntry& entry : entries)
    {
        writePod(stream, entry.offsetSector);
        writePod(stream, entry.sectorCount);
        writePod(stream, entry.byteSize);
        writePod(stream, entry.reserved);
    }
}

void WorldSave::ensureRegionFile(const std::wstring& path)
{
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    if (std::filesystem::exists(std::filesystem::path(path)))
    {
        return;
    }

    std::ofstream stream(std::filesystem::path(path), std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Failed to create region file.");
    }
    std::vector<char> header(kHeaderSize, 0);
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
}

std::uint32_t WorldSave::appendOffsetSector(const std::wstring& path) const
{
    const std::uintmax_t size = std::filesystem::file_size(std::filesystem::path(path));
    return std::max<std::uint32_t>(
        kHeaderSectors,
        static_cast<std::uint32_t>((size + kSectorSize - 1) / kSectorSize));
}

void WorldSave::appendU16(std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value & 0xff));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void WorldSave::appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value & 0xff));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
    bytes.push_back(static_cast<std::uint8_t>((value >> 16) & 0xff));
    bytes.push_back(static_cast<std::uint8_t>((value >> 24) & 0xff));
}

std::uint16_t WorldSave::readU16(const std::vector<std::uint8_t>& bytes, std::size_t& offset)
{
    if (offset + 2 > bytes.size())
    {
        throw std::runtime_error("RLE stream is truncated.");
    }
    const std::uint16_t value = static_cast<std::uint16_t>(bytes[offset]) |
        (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
    offset += 2;
    return value;
}

std::uint32_t WorldSave::readU32(const std::vector<std::uint8_t>& bytes, std::size_t& offset)
{
    if (offset + 4 > bytes.size())
    {
        throw std::runtime_error("RLE stream is truncated.");
    }
    const std::uint32_t value = static_cast<std::uint32_t>(bytes[offset]) |
        (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
        (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
        (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
    offset += 4;
    return value;
}

std::vector<std::uint8_t> WorldSave::encodeRle(const ChunkVoxelData& voxels)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(voxels.blockIds.size());

    struct Run
    {
        std::uint16_t blockId = kAirBlockId;
        std::uint16_t fluidState = kNoFluidState;
        std::uint32_t count = 0;
    };

    std::vector<Run> runs;
    runs.reserve(1024);
    for (std::size_t i = 0; i < voxels.blockIds.size(); ++i)
    {
        const std::uint16_t blockId = voxels.blockIds[i];
        std::uint16_t fluidState = voxels.fluidStates[i];
        if (blockId != kAirBlockId)
        {
            fluidState = kNoFluidState;
        }

        if (!runs.empty() &&
            runs.back().blockId == blockId &&
            runs.back().fluidState == fluidState)
        {
            ++runs.back().count;
        }
        else
        {
            runs.push_back({blockId, fluidState, 1});
        }
    }

    appendU32(bytes, static_cast<std::uint32_t>(runs.size()));
    for (const Run& run : runs)
    {
        appendU16(bytes, run.blockId);
        appendU16(bytes, run.fluidState);
        appendU32(bytes, run.count);
    }
    return bytes;
}

ChunkVoxelData WorldSave::decodeRle(const std::vector<std::uint8_t>& bytes)
{
    std::size_t offset = 0;
    const std::uint32_t runCount = readU32(bytes, offset);
    ChunkVoxelData voxels{};
    voxels.blockIds.reserve(kChunkBlockCount);
    voxels.fluidStates.reserve(kChunkBlockCount);

    for (std::uint32_t i = 0; i < runCount; ++i)
    {
        const std::uint16_t blockId = readU16(bytes, offset);
        std::uint16_t fluidState = readU16(bytes, offset);
        const std::uint32_t count = readU32(bytes, offset);
        if (count == 0)
        {
            throw std::runtime_error("RLE stream contains an empty run.");
        }
        if (voxels.blockIds.size() + count > kChunkBlockCount)
        {
            throw std::runtime_error("RLE stream expands past chunk size.");
        }
        if (blockId != kAirBlockId)
        {
            fluidState = kNoFluidState;
        }
        voxels.blockIds.insert(voxels.blockIds.end(), count, blockId);
        voxels.fluidStates.insert(voxels.fluidStates.end(), count, fluidState);
    }

    if (!voxels.valid())
    {
        throw std::runtime_error("RLE stream did not fill a chunk.");
    }
    if (offset != bytes.size())
    {
        throw std::runtime_error("RLE stream has trailing bytes.");
    }
    return voxels;
}

std::vector<std::uint8_t> WorldSave::compressLz4(const std::vector<std::uint8_t>& bytes)
{
    const int maxCompressedSize = LZ4_compressBound(static_cast<int>(bytes.size()));
    std::vector<std::uint8_t> compressed(static_cast<std::size_t>(maxCompressedSize));
    const int compressedSize = LZ4_compress_default(
        reinterpret_cast<const char*>(bytes.data()),
        reinterpret_cast<char*>(compressed.data()),
        static_cast<int>(bytes.size()),
        maxCompressedSize);
    if (compressedSize <= 0)
    {
        throw std::runtime_error("LZ4 compression failed.");
    }
    compressed.resize(static_cast<std::size_t>(compressedSize));
    return compressed;
}

std::vector<std::uint8_t> WorldSave::decompressLz4(const std::vector<std::uint8_t>& bytes)
{
    std::vector<std::uint8_t> decompressed(kMaxRleBytes);
    const int decompressedSize = LZ4_decompress_safe(
        reinterpret_cast<const char*>(bytes.data()),
        reinterpret_cast<char*>(decompressed.data()),
        static_cast<int>(bytes.size()),
        static_cast<int>(decompressed.size()));
    if (decompressedSize < 0)
    {
        throw std::runtime_error("LZ4 decompression failed.");
    }
    decompressed.resize(static_cast<std::size_t>(decompressedSize));
    return decompressed;
}
