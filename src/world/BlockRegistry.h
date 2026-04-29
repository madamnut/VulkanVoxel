#pragma once

#include "world/Block.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class BlockRegistry
{
public:
    BlockRegistry();

    void loadFromConfig(const std::string& configPath);
    void addDefinition(std::uint16_t id, std::string name, bool solid);
    void addDefinition(
        std::uint16_t id,
        std::string name,
        BlockRenderShape renderShape,
        BlockRenderLayer renderLayer,
        bool collision,
        bool raycast,
        bool faceOccluder,
        bool aoOccluder);

    const BlockDefinition* definitionForId(std::uint16_t blockId) const;
    bool isSolid(std::uint16_t blockId) const;
    bool hasRenderableShape(std::uint16_t blockId) const;
    bool isCollision(std::uint16_t blockId) const;
    bool isRaycastTarget(std::uint16_t blockId) const;
    bool isFaceOccluder(std::uint16_t blockId) const;
    bool isAoOccluder(std::uint16_t blockId) const;
    BlockRenderShape renderShape(std::uint16_t blockId) const;

    const std::vector<BlockDefinition>& definitions() const;
    std::vector<BlockDefinition>& mutableDefinitions();

private:
    std::vector<BlockDefinition> definitions_;
    std::array<int, kBlockIdCount> definitionIndices_{};
};
