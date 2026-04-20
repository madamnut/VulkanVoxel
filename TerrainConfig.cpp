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

}  // namespace

TerrainConfig LoadTerrainConfig(const std::string& path) {
    const std::string json = ReadTextFile(path);

    TerrainConfig config{};
    config.seed = ReadJsonInt(json, "seed", 1337);
    config.wrapSizeXZ = ReadJsonInt(json, "wrap_size_xz", 65536);
    config.baseHeight = ReadJsonFloat(json, "base_height", 192.0f);
    config.solidThreshold = ReadJsonFloat(json, "solid_threshold", 0.0f);
    config.gradientStrength = ReadJsonFloat(json, "gradient_strength", 1.0f);
    config.densityAmplitude = ReadJsonFloat(json, "density_amplitude", 32.0f);
    config.featureScaleXZ = ReadJsonFloat(json, "feature_scale_xz", 128.0f);
    config.featureScaleY = ReadJsonFloat(json, "feature_scale_y", 96.0f);

    config.fbmGain = ReadJsonFloat(json, "fbm_gain", 0.5f);
    config.fbmOctaves = ReadJsonInt(json, "fbm_octaves", 4);
    config.fbmLacunarity = ReadJsonFloat(json, "fbm_lacunarity", 2.0f);

    return config;
}
