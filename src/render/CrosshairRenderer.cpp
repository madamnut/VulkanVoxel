#include "render/CrosshairRenderer.h"

#include <array>

std::vector<DebugTextVertex> buildCrosshairVertices(
    std::uint32_t textureWidth,
    std::uint32_t textureHeight,
    std::uint32_t viewportWidth,
    std::uint32_t viewportHeight)
{
    if (textureWidth == 0 || textureHeight == 0 || viewportWidth == 0 || viewportHeight == 0)
    {
        return {};
    }

    const float width = static_cast<float>(textureWidth);
    const float height = static_cast<float>(textureHeight);
    const float left = static_cast<float>(viewportWidth) * 0.5f - width * 0.5f;
    const float top = static_cast<float>(viewportHeight) * 0.5f - height * 0.5f;
    const float x0 = -1.0f + (2.0f * left / static_cast<float>(viewportWidth));
    const float y0 = 1.0f - (2.0f * top / static_cast<float>(viewportHeight));
    const float x1 = -1.0f + (2.0f * (left + width) / static_cast<float>(viewportWidth));
    const float y1 = 1.0f - (2.0f * (top + height) / static_cast<float>(viewportHeight));

    const auto makeVertex = [](float px, float py, float u, float v) -> DebugTextVertex
    {
        return {{px, py}, {u, v}, {1.0f, 1.0f, 1.0f, 1.0f}};
    };

    const std::array<DebugTextVertex, kCrosshairVertexCount> vertices = {
        makeVertex(x0, y0, 0.0f, 0.0f),
        makeVertex(x1, y0, 1.0f, 0.0f),
        makeVertex(x1, y1, 1.0f, 1.0f),
        makeVertex(x0, y0, 0.0f, 0.0f),
        makeVertex(x1, y1, 1.0f, 1.0f),
        makeVertex(x0, y1, 0.0f, 1.0f),
    };

    return {vertices.begin(), vertices.end()};
}
