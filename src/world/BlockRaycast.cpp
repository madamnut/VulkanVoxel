#include "world/BlockRaycast.h"

#include <cmath>
#include <limits>

namespace
{
float intBound(float origin, float direction)
{
    if (direction > 0.0f)
    {
        return (std::floor(origin + 1.0f) - origin) / direction;
    }
    if (direction < 0.0f)
    {
        return (origin - std::floor(origin)) / -direction;
    }
    return std::numeric_limits<float>::infinity();
}
}

std::optional<BlockRaycastHit> raycastBlocks(
    Vec3 origin,
    Vec3 direction,
    float maxDistance,
    const RaycastSolidBlockQuery& isSolidBlock,
    const RaycastTraversableBlockQuery& canTraverseBlock)
{
    direction = normalize(direction);
    if (length(direction) <= 0.00001f || maxDistance <= 0.0f)
    {
        return std::nullopt;
    }

    int blockX = static_cast<int>(std::floor(origin.x));
    int blockY = static_cast<int>(std::floor(origin.y));
    int blockZ = static_cast<int>(std::floor(origin.z));

    const int stepX = direction.x > 0.0f ? 1 : (direction.x < 0.0f ? -1 : 0);
    const int stepY = direction.y > 0.0f ? 1 : (direction.y < 0.0f ? -1 : 0);
    const int stepZ = direction.z > 0.0f ? 1 : (direction.z < 0.0f ? -1 : 0);

    const float deltaX = stepX != 0 ? std::abs(1.0f / direction.x) : std::numeric_limits<float>::infinity();
    const float deltaY = stepY != 0 ? std::abs(1.0f / direction.y) : std::numeric_limits<float>::infinity();
    const float deltaZ = stepZ != 0 ? std::abs(1.0f / direction.z) : std::numeric_limits<float>::infinity();

    float tMaxX = intBound(origin.x, direction.x);
    float tMaxY = intBound(origin.y, direction.y);
    float tMaxZ = intBound(origin.z, direction.z);
    float traveled = 0.0f;
    int normalX = 0;
    int normalY = 0;
    int normalZ = 0;

    while (traveled <= maxDistance)
    {
        if (!canTraverseBlock(blockX, blockY, blockZ))
        {
            return std::nullopt;
        }
        if (isSolidBlock(blockX, blockY, blockZ))
        {
            return BlockRaycastHit{blockX, blockY, blockZ, normalX, normalY, normalZ, traveled};
        }

        if (tMaxX < tMaxY)
        {
            if (tMaxX < tMaxZ)
            {
                blockX += stepX;
                traveled = tMaxX;
                tMaxX += deltaX;
                normalX = -stepX;
                normalY = 0;
                normalZ = 0;
            }
            else
            {
                blockZ += stepZ;
                traveled = tMaxZ;
                tMaxZ += deltaZ;
                normalX = 0;
                normalY = 0;
                normalZ = -stepZ;
            }
        }
        else
        {
            if (tMaxY < tMaxZ)
            {
                blockY += stepY;
                traveled = tMaxY;
                tMaxY += deltaY;
                normalX = 0;
                normalY = -stepY;
                normalZ = 0;
            }
            else
            {
                blockZ += stepZ;
                traveled = tMaxZ;
                tMaxZ += deltaZ;
                normalX = 0;
                normalY = 0;
                normalZ = -stepZ;
            }
        }
    }

    return std::nullopt;
}
