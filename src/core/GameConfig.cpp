#include "core/GameConfig.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>

namespace
{
int readInt(const std::string& config, const std::string& key, int fallback)
{
    const std::string quotedKey = "\"" + key + "\"";
    const std::size_t keyPosition = config.find(quotedKey);
    if (keyPosition == std::string::npos)
    {
        return fallback;
    }

    const std::size_t colonPosition = config.find(':', keyPosition + quotedKey.size());
    if (colonPosition == std::string::npos)
    {
        return fallback;
    }

    const std::size_t valueStart = config.find_first_of("-0123456789", colonPosition + 1);
    if (valueStart == std::string::npos)
    {
        return fallback;
    }

    const std::size_t valueEnd = config.find_first_not_of("0123456789", valueStart + 1);
    try
    {
        return std::stoi(config.substr(valueStart, valueEnd - valueStart));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}
}

WorldConfig loadWorldConfigFile(
    const std::string& path,
    WorldConfig defaults,
    WorldConfig minimums,
    WorldConfig maximums)
{
    std::ifstream configFile(path);
    if (!configFile)
    {
        return defaults;
    }

    const std::string config(
        (std::istreambuf_iterator<char>(configFile)),
        std::istreambuf_iterator<char>());

    return {
        std::clamp(
            readInt(config, "chunkLoadRadius", defaults.chunkLoadRadius),
            minimums.chunkLoadRadius,
            maximums.chunkLoadRadius),
        std::clamp(
            readInt(config, "chunkUploadsPerFrame", defaults.chunkUploadsPerFrame),
            minimums.chunkUploadsPerFrame,
            maximums.chunkUploadsPerFrame),
        std::clamp(
            readInt(config, "chunkBuildThreads", defaults.chunkBuildThreads),
            minimums.chunkBuildThreads,
            maximums.chunkBuildThreads),
    };
}
