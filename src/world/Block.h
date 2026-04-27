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
constexpr std::uint16_t kBedrockBlockId = std::numeric_limits<std::uint16_t>::max();

enum class BlockFace : std::uint8_t
{
    Top = 0,
    Side = 1,
    Bottom = 2,
};

struct BlockDefinition
{
    std::uint16_t id = kAirBlockId;
    std::string name;
    bool solid = false;
    std::array<std::uint32_t, 3> textureLayers{};
};
