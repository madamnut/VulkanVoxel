#pragma once

#include "core/Math.h"
#include "world/ChunkTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class PlayerModel
{
public:
    void loadFromFile(const std::string& path);

    bool isLoaded() const;
    std::size_t vertexCount() const;
    std::size_t indexCount() const;
    const std::vector<std::uint32_t>& indices() const;

    const std::vector<BlockVertex>& updateRenderVertices(Vec3 feetPosition, float yaw, float playerHeight);

private:
    std::vector<BlockVertex> baseVertices_;
    std::vector<BlockVertex> renderVertices_;
    std::vector<std::uint32_t> indices_;
    Vec3 boundsMin_{};
    Vec3 boundsMax_{};
    bool loaded_ = false;
};
