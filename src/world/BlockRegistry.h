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

    const BlockDefinition* definitionForId(std::uint16_t blockId) const;
    bool isSolid(std::uint16_t blockId) const;

    const std::vector<BlockDefinition>& definitions() const;
    std::vector<BlockDefinition>& mutableDefinitions();

private:
    std::vector<BlockDefinition> definitions_;
    std::array<int, kBlockIdCount> definitionIndices_{};
};
