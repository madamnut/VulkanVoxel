#pragma once

#include <array>
#include <cstdint>
#include <string_view>

using BlockId = std::uint16_t;

enum class BlockFace : std::uint8_t {
    Top = 0,
    Bottom,
    Left,
    Right,
    Front,
    Back,
    Count
};

constexpr BlockId kBlockAir = 0;
constexpr BlockId kBlockRock = 1;
constexpr BlockId kBlockDirt = 2;
constexpr BlockId kBlockGrass = 3;

constexpr std::uint16_t kBlockTextureLayerRock = 0;
constexpr std::uint16_t kBlockTextureLayerDirt = 1;
constexpr std::uint16_t kBlockTextureLayerGrassTop = 2;
constexpr std::uint16_t kBlockTextureLayerGrassSide = 3;
constexpr std::uint16_t kBlockTextureLayerGrassBottom = 4;

inline constexpr std::array<std::string_view, 5> kBlockTextureLayerFiles = {
    "rock.png",
    "dirt.png",
    "grass_top.png",
    "grass_side.png",
    "grass_bottom.png",
};

constexpr BlockFace GetBlockFaceForAxis(int axis, bool positiveNormal) {
    switch (axis) {
    case 0:
        return positiveNormal ? BlockFace::Right : BlockFace::Left;
    case 1:
        return positiveNormal ? BlockFace::Top : BlockFace::Bottom;
    case 2:
    default:
        return positiveNormal ? BlockFace::Front : BlockFace::Back;
    }
}

constexpr std::uint16_t GetBlockTextureLayer(BlockId blockId, BlockFace face) {
    switch (blockId) {
    case kBlockGrass:
        if (face == BlockFace::Top) {
            return kBlockTextureLayerGrassTop;
        }
        if (face == BlockFace::Bottom) {
            return kBlockTextureLayerGrassBottom;
        }
        return kBlockTextureLayerGrassSide;
    case kBlockDirt:
        return kBlockTextureLayerDirt;
    case kBlockRock:
    default:
        return kBlockTextureLayerRock;
    }
}
