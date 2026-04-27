#pragma once

#include "core/Math.h"

#include <functional>
#include <optional>

struct BlockRaycastHit
{
    int x = 0;
    int y = 0;
    int z = 0;
    int normalX = 0;
    int normalY = 0;
    int normalZ = 0;
    float distance = 0.0f;
};

using RaycastSolidBlockQuery = std::function<bool(int, int, int)>;
using RaycastTraversableBlockQuery = std::function<bool(int, int, int)>;

std::optional<BlockRaycastHit> raycastBlocks(
    Vec3 origin,
    Vec3 direction,
    float maxDistance,
    const RaycastSolidBlockQuery& isSolidBlock,
    const RaycastTraversableBlockQuery& canTraverseBlock);
