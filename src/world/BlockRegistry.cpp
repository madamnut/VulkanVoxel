#include "world/BlockRegistry.h"

#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace
{
std::optional<std::uint32_t> readJsonUInt(const std::string& object, const char* key)
{
    const std::string pattern = "\"" + std::string(key) + "\"";
    const std::size_t keyPos = object.find(pattern);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }

    const std::size_t colonPos = object.find(':', keyPos + pattern.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }

    std::size_t valuePos = colonPos + 1;
    while (valuePos < object.size() && std::isspace(static_cast<unsigned char>(object[valuePos])))
    {
        ++valuePos;
    }

    std::size_t endPos = valuePos;
    while (endPos < object.size() && std::isdigit(static_cast<unsigned char>(object[endPos])))
    {
        ++endPos;
    }
    if (endPos == valuePos)
    {
        return std::nullopt;
    }

    return static_cast<std::uint32_t>(std::stoul(object.substr(valuePos, endPos - valuePos)));
}

std::optional<std::string> readJsonString(const std::string& object, const char* key)
{
    const std::string pattern = "\"" + std::string(key) + "\"";
    const std::size_t keyPos = object.find(pattern);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }

    const std::size_t colonPos = object.find(':', keyPos + pattern.size());
    const std::size_t quotePos = object.find('"', colonPos == std::string::npos ? keyPos : colonPos);
    if (colonPos == std::string::npos || quotePos == std::string::npos)
    {
        return std::nullopt;
    }

    const std::size_t endQuotePos = object.find('"', quotePos + 1);
    if (endQuotePos == std::string::npos)
    {
        return std::nullopt;
    }
    return object.substr(quotePos + 1, endQuotePos - quotePos - 1);
}

bool readJsonBool(const std::string& object, const char* key, bool fallback)
{
    const std::string pattern = "\"" + std::string(key) + "\"";
    const std::size_t keyPos = object.find(pattern);
    if (keyPos == std::string::npos)
    {
        return fallback;
    }

    const std::size_t colonPos = object.find(':', keyPos + pattern.size());
    if (colonPos == std::string::npos)
    {
        return fallback;
    }

    std::size_t valuePos = colonPos + 1;
    while (valuePos < object.size() && std::isspace(static_cast<unsigned char>(object[valuePos])))
    {
        ++valuePos;
    }
    return object.compare(valuePos, 4, "true") == 0;
}

std::optional<bool> readJsonBoolOptional(const std::string& object, const char* key)
{
    const std::string pattern = "\"" + std::string(key) + "\"";
    const std::size_t keyPos = object.find(pattern);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }

    const std::size_t colonPos = object.find(':', keyPos + pattern.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }

    std::size_t valuePos = colonPos + 1;
    while (valuePos < object.size() && std::isspace(static_cast<unsigned char>(object[valuePos])))
    {
        ++valuePos;
    }
    if (object.compare(valuePos, 4, "true") == 0)
    {
        return true;
    }
    if (object.compare(valuePos, 5, "false") == 0)
    {
        return false;
    }
    return std::nullopt;
}

BlockRenderShape parseRenderShape(const std::optional<std::string>& value, bool solidFallback)
{
    if (!value)
    {
        return solidFallback ? BlockRenderShape::Cube : BlockRenderShape::None;
    }
    if (*value == "cube")
    {
        return BlockRenderShape::Cube;
    }
    if (*value == "cross")
    {
        return BlockRenderShape::Cross;
    }
    return BlockRenderShape::None;
}

BlockRenderLayer parseRenderLayer(const std::optional<std::string>& value)
{
    if (!value || *value == "opaque")
    {
        return BlockRenderLayer::Opaque;
    }
    if (*value == "cutout")
    {
        return BlockRenderLayer::Cutout;
    }
    if (*value == "blend")
    {
        return BlockRenderLayer::Blend;
    }
    return BlockRenderLayer::Opaque;
}
}

BlockRegistry::BlockRegistry()
{
    definitionIndices_.fill(-1);
}

void BlockRegistry::loadFromConfig(const std::string& configPath)
{
    definitions_.clear();
    definitionIndices_.fill(-1);

    std::ifstream configFile(configPath);
    if (configFile)
    {
        const std::string config(
            (std::istreambuf_iterator<char>(configFile)),
            std::istreambuf_iterator<char>());
        const std::size_t blocksPos = config.find("\"blocks\"");
        const std::size_t arrayBegin = config.find('[', blocksPos);
        const std::size_t arrayEnd = config.find(']', arrayBegin);
        std::size_t cursor = arrayBegin;
        while (cursor != std::string::npos && cursor < arrayEnd)
        {
            const std::size_t objectBegin = config.find('{', cursor);
            if (objectBegin == std::string::npos || objectBegin > arrayEnd)
            {
                break;
            }
            const std::size_t objectEnd = config.find('}', objectBegin);
            if (objectEnd == std::string::npos || objectEnd > arrayEnd)
            {
                break;
            }

            const std::string object = config.substr(objectBegin, objectEnd - objectBegin + 1);
            const auto id = readJsonUInt(object, "id");
            const auto name = readJsonString(object, "name");
            if (id && *id <= std::numeric_limits<std::uint16_t>::max() && name && !name->empty())
            {
                const bool solidFallback = readJsonBool(object, "solid", true);
                const BlockRenderShape renderShape =
                    parseRenderShape(readJsonString(object, "renderShape"), solidFallback);
                const BlockRenderLayer renderLayer =
                    parseRenderLayer(readJsonString(object, "renderLayer"));
                const bool collision = readJsonBoolOptional(object, "collision").value_or(solidFallback);
                const bool raycast = readJsonBoolOptional(object, "raycast").value_or(
                    collision || renderShape != BlockRenderShape::None);
                const bool faceOccluder = readJsonBoolOptional(object, "faceOccluder").value_or(
                    solidFallback && renderShape == BlockRenderShape::Cube);
                const bool aoOccluder = readJsonBoolOptional(object, "aoOccluder").value_or(faceOccluder);
                addDefinition(
                    static_cast<std::uint16_t>(*id),
                    *name,
                    renderShape,
                    renderLayer,
                    collision,
                    raycast,
                    faceOccluder,
                    aoOccluder);
            }

            cursor = objectEnd + 1;
        }
    }

    if (definitions_.empty())
    {
        addDefinition(kRockBlockId, "rock", true);
        addDefinition(kDirtBlockId, "dirt", true);
        addDefinition(kGrassBlockId, "grass", true);
        addDefinition(kTrunkBlockId, "trunk", true);
        addDefinition(
            kLeavesBlockId,
            "leaves",
            BlockRenderShape::Cube,
            BlockRenderLayer::Cutout,
            true,
            true,
            false,
            false);
        addDefinition(kClayBlockId, "clay", true);
        addDefinition(kMudBlockId, "mud", true);
        addDefinition(kSandBlockId, "sand", true);
        addDefinition(kSandstoneBlockId, "sandstone", true);
        addDefinition(
            kPlantBlockId,
            "plant",
            BlockRenderShape::Cross,
            BlockRenderLayer::Cutout,
            false,
            true,
            false,
            false);
        addDefinition(kBedrockBlockId, "bedrock", true);
    }
}

void BlockRegistry::addDefinition(std::uint16_t id, std::string name, bool solid)
{
    addDefinition(
        id,
        std::move(name),
        solid ? BlockRenderShape::Cube : BlockRenderShape::None,
        BlockRenderLayer::Opaque,
        solid,
        solid,
        solid,
        solid);
}

void BlockRegistry::addDefinition(
    std::uint16_t id,
    std::string name,
    BlockRenderShape renderShape,
    BlockRenderLayer renderLayer,
    bool collision,
    bool raycast,
    bool faceOccluder,
    bool aoOccluder)
{
    BlockDefinition definition{};
    definition.id = id;
    definition.name = std::move(name);
    definition.solid = collision;
    definition.renderShape = renderShape;
    definition.renderLayer = renderLayer;
    definition.collision = collision;
    definition.raycast = raycast;
    definition.faceOccluder = faceOccluder;
    definition.aoOccluder = aoOccluder;
    definitionIndices_[id] = static_cast<int>(definitions_.size());
    definitions_.push_back(std::move(definition));
}

const BlockDefinition* BlockRegistry::definitionForId(std::uint16_t blockId) const
{
    const int index = definitionIndices_[blockId];
    if (index < 0)
    {
        return nullptr;
    }
    return &definitions_[static_cast<std::size_t>(index)];
}

bool BlockRegistry::isSolid(std::uint16_t blockId) const
{
    return isCollision(blockId);
}

bool BlockRegistry::hasRenderableShape(std::uint16_t blockId) const
{
    return renderShape(blockId) != BlockRenderShape::None;
}

bool BlockRegistry::isCollision(std::uint16_t blockId) const
{
    const BlockDefinition* definition = definitionForId(blockId);
    return definition != nullptr && definition->collision;
}

bool BlockRegistry::isRaycastTarget(std::uint16_t blockId) const
{
    const BlockDefinition* definition = definitionForId(blockId);
    return definition != nullptr && definition->raycast;
}

bool BlockRegistry::isFaceOccluder(std::uint16_t blockId) const
{
    const BlockDefinition* definition = definitionForId(blockId);
    return definition != nullptr && definition->faceOccluder;
}

bool BlockRegistry::isAoOccluder(std::uint16_t blockId) const
{
    const BlockDefinition* definition = definitionForId(blockId);
    return definition != nullptr && definition->aoOccluder;
}

BlockRenderShape BlockRegistry::renderShape(std::uint16_t blockId) const
{
    const BlockDefinition* definition = definitionForId(blockId);
    return definition != nullptr ? definition->renderShape : BlockRenderShape::None;
}

const std::vector<BlockDefinition>& BlockRegistry::definitions() const
{
    return definitions_;
}

std::vector<BlockDefinition>& BlockRegistry::mutableDefinitions()
{
    return definitions_;
}
