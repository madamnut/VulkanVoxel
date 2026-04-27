#include "render/PlayerModel.h"

#include "core/Math.h"
#include "render/VulkanHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

void PlayerModel::loadFromFile(const std::string& path)
{
    const std::vector<char> data = readBinaryFile(path);
    constexpr std::size_t headerSize = 12;
    if (data.size() < headerSize || std::memcmp(data.data(), "PMSH", 4) != 0)
    {
        throw std::runtime_error("Invalid extracted player mesh.");
    }

    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::memcpy(&vertexCount, data.data() + 4, sizeof(vertexCount));
    std::memcpy(&indexCount, data.data() + 8, sizeof(indexCount));

    constexpr std::size_t serializedVertexSize = sizeof(float) * 5;
    const std::size_t expectedSize =
        headerSize +
        static_cast<std::size_t>(vertexCount) * serializedVertexSize +
        static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t);
    if (vertexCount == 0 || indexCount == 0 || data.size() < expectedSize)
    {
        throw std::runtime_error("Extracted player mesh is empty or truncated.");
    }

    baseVertices_.resize(vertexCount);
    renderVertices_.resize(vertexCount);
    std::size_t offset = headerSize;
    for (std::uint32_t i = 0; i < vertexCount; ++i)
    {
        float serialized[5]{};
        std::memcpy(serialized, data.data() + offset, sizeof(serialized));
        offset += sizeof(serialized);

        BlockVertex& vertex = baseVertices_[i];
        vertex.position[0] = serialized[0];
        vertex.position[1] = serialized[1];
        vertex.position[2] = serialized[2];
        vertex.uv[0] = serialized[3];
        vertex.uv[1] = serialized[4];
        vertex.ao = 1.0f;
        vertex.textureLayer = 0.0f;
    }

    indices_.resize(indexCount);
    std::memcpy(indices_.data(), data.data() + offset, static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t));

    boundsMin_ = {
        baseVertices_[0].position[0],
        baseVertices_[0].position[1],
        baseVertices_[0].position[2],
    };
    boundsMax_ = boundsMin_;
    for (const BlockVertex& vertex : baseVertices_)
    {
        boundsMin_.x = std::min(boundsMin_.x, vertex.position[0]);
        boundsMin_.y = std::min(boundsMin_.y, vertex.position[1]);
        boundsMin_.z = std::min(boundsMin_.z, vertex.position[2]);
        boundsMax_.x = std::max(boundsMax_.x, vertex.position[0]);
        boundsMax_.y = std::max(boundsMax_.y, vertex.position[1]);
        boundsMax_.z = std::max(boundsMax_.z, vertex.position[2]);
    }

    loaded_ = true;
}

bool PlayerModel::isLoaded() const
{
    return loaded_;
}

std::size_t PlayerModel::vertexCount() const
{
    return baseVertices_.size();
}

std::size_t PlayerModel::indexCount() const
{
    return indices_.size();
}

const std::vector<std::uint32_t>& PlayerModel::indices() const
{
    return indices_;
}

const std::vector<BlockVertex>& PlayerModel::updateRenderVertices(Vec3 feetPosition, float yaw, float playerHeight)
{
    const float modelHeight = std::max(boundsMax_.y - boundsMin_.y, 0.0001f);
    const float scale = playerHeight / modelHeight;
    const float centerX = (boundsMin_.x + boundsMax_.x) * 0.5f;
    const float centerZ = (boundsMin_.z + boundsMax_.z) * 0.5f;
    const float yawRadians = yaw + kPi * 0.5f;
    const float cosYaw = std::cos(yawRadians);
    const float sinYaw = std::sin(yawRadians);

    renderVertices_.resize(baseVertices_.size());
    for (std::size_t i = 0; i < baseVertices_.size(); ++i)
    {
        const BlockVertex& source = baseVertices_[i];
        BlockVertex& destination = renderVertices_[i];

        const float localX = (source.position[0] - centerX) * scale;
        const float localY = (source.position[1] - boundsMin_.y) * scale;
        const float localZ = (source.position[2] - centerZ) * scale;
        const float rotatedX = localX * cosYaw - localZ * sinYaw;
        const float rotatedZ = localX * sinYaw + localZ * cosYaw;

        destination.position[0] = feetPosition.x + rotatedX;
        destination.position[1] = feetPosition.y + localY;
        destination.position[2] = feetPosition.z + rotatedZ;
        destination.uv[0] = source.uv[0];
        destination.uv[1] = source.uv[1];
        destination.ao = source.ao;
        destination.textureLayer = 0.0f;
    }

    return renderVertices_;
}
