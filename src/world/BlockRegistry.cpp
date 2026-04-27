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
                addDefinition(
                    static_cast<std::uint16_t>(*id),
                    *name,
                    readJsonBool(object, "solid", true));
            }

            cursor = objectEnd + 1;
        }
    }

    if (definitions_.empty())
    {
        addDefinition(kRockBlockId, "rock", true);
        addDefinition(kDirtBlockId, "dirt", true);
        addDefinition(kGrassBlockId, "grass", true);
        addDefinition(kBedrockBlockId, "bedrock", true);
    }
}

void BlockRegistry::addDefinition(std::uint16_t id, std::string name, bool solid)
{
    BlockDefinition definition{};
    definition.id = id;
    definition.name = std::move(name);
    definition.solid = solid;
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
    const BlockDefinition* definition = definitionForId(blockId);
    return definition != nullptr && definition->solid;
}

const std::vector<BlockDefinition>& BlockRegistry::definitions() const
{
    return definitions_;
}

std::vector<BlockDefinition>& BlockRegistry::mutableDefinitions()
{
    return definitions_;
}
