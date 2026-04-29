#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

constexpr std::size_t kBlockIdCount =
    static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1;
constexpr std::uint16_t kAirBlockId = 0;
constexpr std::uint16_t kRockBlockId = 1;
constexpr std::uint16_t kDirtBlockId = 2;
constexpr std::uint16_t kGrassBlockId = 3;
constexpr std::uint16_t kTrunkBlockId = 4;
constexpr std::uint16_t kLeavesBlockId = 5;
constexpr std::uint16_t kClayBlockId = 6;
constexpr std::uint16_t kMudBlockId = 7;
constexpr std::uint16_t kSandBlockId = 8;
constexpr std::uint16_t kSandstoneBlockId = 9;
constexpr std::uint16_t kPlantBlockId = 10000;
constexpr std::uint16_t kBedrockBlockId = std::numeric_limits<std::uint16_t>::max();

enum class BlockFace : std::uint8_t
{
    Top = 0,
    Side = 1,
    Bottom = 2,
};

enum class BlockRenderShape : std::uint8_t
{
    None = 0,
    Cube = 1,
    Cross = 2,
};

enum class BlockRenderLayer : std::uint8_t
{
    Opaque = 0,
    Cutout = 1,
    Blend = 2,
};

struct BlockDefinition
{
    std::uint16_t id = kAirBlockId;
    std::string name;
    bool solid = false; // Legacy fallback for old block config entries.
    BlockRenderShape renderShape = BlockRenderShape::None;
    BlockRenderLayer renderLayer = BlockRenderLayer::Opaque;
    bool collision = false;
    bool raycast = false;
    bool faceOccluder = false;
    bool aoOccluder = false;
    std::array<std::uint32_t, 3> textureLayers{};
};
