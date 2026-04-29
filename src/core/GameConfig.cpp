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

float readFloat(const std::string& config, const std::string& key, float fallback)
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

    const std::size_t valueStart = config.find_first_of("-0123456789.", colonPosition + 1);
    if (valueStart == std::string::npos)
    {
        return fallback;
    }

    const std::size_t valueEnd = config.find_first_not_of("0123456789+-.eE", valueStart + 1);
    try
    {
        return std::stof(config.substr(valueStart, valueEnd - valueStart));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

TerrainDensityConfig readTerrainDensityConfig(
    const std::string& config,
    TerrainDensityConfig defaults,
    TerrainDensityConfig minimums,
    TerrainDensityConfig maximums)
{
    TerrainDensityConfig result = defaults;
    result.gradient.center = std::clamp(
        readFloat(config, "center", defaults.gradient.center),
        minimums.gradient.center,
        maximums.gradient.center);
    result.gradient.strength = std::clamp(
        readFloat(config, "strength", defaults.gradient.strength),
        minimums.gradient.strength,
        maximums.gradient.strength);
    result.noise.octaves = std::clamp(
        readInt(config, "octaves", defaults.noise.octaves),
        minimums.noise.octaves,
        maximums.noise.octaves);
    result.noise.baseFrequency = std::clamp(
        readInt(config, "baseFrequency", defaults.noise.baseFrequency),
        minimums.noise.baseFrequency,
        maximums.noise.baseFrequency);
    result.noise.frequencyMultiplier = std::clamp(
        readFloat(config, "frequencyMultiplier", defaults.noise.frequencyMultiplier),
        minimums.noise.frequencyMultiplier,
        maximums.noise.frequencyMultiplier);
    result.noise.baseAmplitude = std::clamp(
        readFloat(config, "baseAmplitude", defaults.noise.baseAmplitude),
        minimums.noise.baseAmplitude,
        maximums.noise.baseAmplitude);
    result.noise.amplitudeMultiplier = std::clamp(
        readFloat(config, "amplitudeMultiplier", defaults.noise.amplitudeMultiplier),
        minimums.noise.amplitudeMultiplier,
        maximums.noise.amplitudeMultiplier);
    result.noise.verticalFrequencyScale = std::clamp(
        readFloat(config, "verticalFrequencyScale", defaults.noise.verticalFrequencyScale),
        minimums.noise.verticalFrequencyScale,
        maximums.noise.verticalFrequencyScale);

    return result;
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

    WorldConfig result{};
    result.chunkLoadRadius = std::clamp(
        readInt(config, "chunkLoadRadius", defaults.chunkLoadRadius),
        minimums.chunkLoadRadius,
        maximums.chunkLoadRadius);
    result.chunkUploadsPerFrame = std::clamp(
        readInt(config, "chunkUploadsPerFrame", defaults.chunkUploadsPerFrame),
        minimums.chunkUploadsPerFrame,
        maximums.chunkUploadsPerFrame);
    result.chunkBuildThreads = std::clamp(
        readInt(config, "chunkBuildThreads", defaults.chunkBuildThreads),
        minimums.chunkBuildThreads,
        maximums.chunkBuildThreads);
    result.terrainDensity = readTerrainDensityConfig(
        config,
        defaults.terrainDensity,
        minimums.terrainDensity,
        maximums.terrainDensity);
    return result;
}
