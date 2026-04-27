#pragma once

#include "render/DebugTextRenderer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr std::size_t kCrosshairVertexCount = 6;

std::vector<DebugTextVertex> buildCrosshairVertices(
    std::uint32_t textureWidth,
    std::uint32_t textureHeight,
    std::uint32_t viewportWidth,
    std::uint32_t viewportHeight);
