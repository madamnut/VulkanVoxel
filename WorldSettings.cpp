#include "WorldSettings.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::string ReadTextFile(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open world settings: " + path);
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

int ReadJsonInt(const std::string& json, const std::string& key, int defaultValue) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return defaultValue;
    }

    return std::stoi(match[1].str());
}

}  // namespace

WorldSettings LoadWorldSettings(const std::string& path) {
    const std::string json = ReadTextFile(path);

    WorldSettings settings{};
    settings.chunkRadius = ReadJsonInt(json, "chunk_radius", 5);
    return settings;
}
