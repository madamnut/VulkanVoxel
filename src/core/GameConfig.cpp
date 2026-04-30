#include "core/GameConfig.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>

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

bool readBool(const std::string& config, const std::string& key, bool fallback)
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

    const std::size_t valueStart = config.find_first_not_of(" \t\r\n", colonPosition + 1);
    if (valueStart == std::string::npos)
    {
        return fallback;
    }
    if (config.compare(valueStart, 4, "true") == 0)
    {
        return true;
    }
    if (config.compare(valueStart, 5, "false") == 0)
    {
        return false;
    }
    return fallback;
}

std::string readObject(const std::string& config, const std::string& key)
{
    const std::string quotedKey = "\"" + key + "\"";
    const std::size_t keyPosition = config.find(quotedKey);
    if (keyPosition == std::string::npos)
    {
        return {};
    }

    const std::size_t objectBegin = config.find('{', keyPosition + quotedKey.size());
    if (objectBegin == std::string::npos)
    {
        return {};
    }

    int depth = 0;
    for (std::size_t cursor = objectBegin; cursor < config.size(); ++cursor)
    {
        if (config[cursor] == '{')
        {
            ++depth;
        }
        else if (config[cursor] == '}')
        {
            --depth;
            if (depth == 0)
            {
                return config.substr(objectBegin, cursor - objectBegin + 1);
            }
        }
    }

    return {};
}

TerrainDensityConfig readTerrainDensityConfig(
    const std::string& config,
    TerrainDensityConfig defaults,
    TerrainDensityConfig minimums,
    TerrainDensityConfig maximums)
{
    TerrainDensityConfig result = defaults;
    const std::string gradientConfig = readObject(config, "gradient");
    const std::string noiseConfig = readObject(config, "noise");
    const std::string landformConfig = readObject(config, "landform");
    const std::string emptyObject = "{}";
    const std::string& gradientSource = gradientConfig.empty() ? config : gradientConfig;
    const std::string& noiseSource = noiseConfig.empty() ? config : noiseConfig;
    const std::string& landformSource = landformConfig.empty() ? emptyObject : landformConfig;

    result.gradient.center = std::clamp(
        readFloat(gradientSource, "center", defaults.gradient.center),
        minimums.gradient.center,
        maximums.gradient.center);
    result.gradient.strength = std::clamp(
        readFloat(gradientSource, "strength", defaults.gradient.strength),
        minimums.gradient.strength,
        maximums.gradient.strength);
    result.noise.octaves = std::clamp(
        readInt(noiseSource, "octaves", defaults.noise.octaves),
        minimums.noise.octaves,
        maximums.noise.octaves);
    result.noise.baseFrequency = std::clamp(
        readInt(noiseSource, "baseFrequency", defaults.noise.baseFrequency),
        minimums.noise.baseFrequency,
        maximums.noise.baseFrequency);
    result.noise.frequencyMultiplier = std::clamp(
        readFloat(noiseSource, "frequencyMultiplier", defaults.noise.frequencyMultiplier),
        minimums.noise.frequencyMultiplier,
        maximums.noise.frequencyMultiplier);
    result.noise.baseAmplitude = std::clamp(
        readFloat(noiseSource, "baseAmplitude", defaults.noise.baseAmplitude),
        minimums.noise.baseAmplitude,
        maximums.noise.baseAmplitude);
    result.noise.amplitudeMultiplier = std::clamp(
        readFloat(noiseSource, "amplitudeMultiplier", defaults.noise.amplitudeMultiplier),
        minimums.noise.amplitudeMultiplier,
        maximums.noise.amplitudeMultiplier);
    result.noise.verticalFrequencyScale = std::clamp(
        readFloat(noiseSource, "verticalFrequencyScale", defaults.noise.verticalFrequencyScale),
        minimums.noise.verticalFrequencyScale,
        maximums.noise.verticalFrequencyScale);
    result.landform.enabled = readBool(landformSource, "enabled", defaults.landform.enabled);
    result.landform.frequency = std::clamp(
        readInt(landformSource, "frequency", defaults.landform.frequency),
        minimums.landform.frequency,
        maximums.landform.frequency);

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
    result.terrainDensity = readTerrainDensityConfig(
        config,
        defaults.terrainDensity,
        minimums.terrainDensity,
        maximums.terrainDensity);
    return result;
}
