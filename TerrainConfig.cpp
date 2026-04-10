#include "TerrainConfig.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::string ReadTextFile(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open terrain config: " + path);
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string EscapeRegexKey(const std::string& key) {
    std::string escaped;
    escaped.reserve(key.size() * 2);

    for (char c : key) {
        switch (c) {
        case '\\':
        case '^':
        case '$':
        case '.':
        case '|':
        case '?':
        case '*':
        case '+':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
            escaped.push_back('\\');
            break;
        default:
            break;
        }
        escaped.push_back(c);
    }

    return escaped;
}

int ReadJsonInt(const std::string& json, const std::string& key, int defaultValue) {
    const std::regex pattern("\"" + EscapeRegexKey(key) + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return defaultValue;
    }

    return std::stoi(match[1].str());
}

float ReadJsonFloat(const std::string& json, const std::string& key, float defaultValue) {
    const std::regex pattern("\"" + EscapeRegexKey(key) + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return defaultValue;
    }

    return std::stof(match[1].str());
}

bool ReadJsonBool(const std::string& json, const std::string& key, bool defaultValue) {
    const std::regex pattern("\"" + EscapeRegexKey(key) + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return defaultValue;
    }

    return match[1].str() == "true";
}

}  // namespace

TerrainConfig LoadTerrainConfig(const std::string& path) {
    const std::string json = ReadTextFile(path);

    TerrainConfig config{};
    config.seed = ReadJsonInt(json, "seed", 1337);

    config.seaLevel = ReadJsonInt(json, "sea_level", 169);
    config.flatGroundHeight = ReadJsonInt(json, "flat_ground_height", 256);
    config.solidThreshold = ReadJsonFloat(json, "solid_threshold", 0.0f);

    config.noiseFeatureScale = ReadJsonFloat(json, "noise_feature_scale", 600.0f);
    config.noiseOutputMin = ReadJsonFloat(json, "noise_output_min", -1.0f);
    config.noiseOutputMax = ReadJsonFloat(json, "noise_output_max", 1.0f);

    config.fbmGain = ReadJsonFloat(json, "fbm_gain", 0.6f);
    config.fbmWeightedStrength = ReadJsonFloat(json, "fbm_weighted_strength", 1.0f);
    config.fbmOctaves = ReadJsonInt(json, "fbm_octaves", 7);
    config.fbmLacunarity = ReadJsonFloat(json, "fbm_lacunarity", 2.5f);

    config.warpEnabled = ReadJsonBool(json, "warp_enabled", true);
    config.warpAmplitude = ReadJsonFloat(json, "warp_amplitude", 60.0f);
    config.warpFeatureScale = ReadJsonFloat(json, "warp_feature_scale", 600.0f);
    config.warpSeedOffset = ReadJsonInt(json, "warp_seed_offset", 0);
    config.warpXAmplitudeScaling = ReadJsonFloat(json, "warp_x_amplitude_scaling", 1.0f);
    config.warpYAmplitudeScaling = ReadJsonFloat(json, "warp_y_amplitude_scaling", 1.0f);
    config.warpZAmplitudeScaling = ReadJsonFloat(json, "warp_z_amplitude_scaling", 1.0f);
    config.warpWAmplitudeScaling = ReadJsonFloat(json, "warp_w_amplitude_scaling", 1.0f);

    config.gradientCenterY = ReadJsonFloat(json, "gradient_center_y", 169.0f);
    config.gradientScale = ReadJsonFloat(json, "gradient_scale", 0.01f);
    config.gradientOffset = ReadJsonFloat(json, "gradient_offset", 0.0f);

    return config;
}
