#pragma once

#include <string>

struct WorldSettings {
    int chunkRadius = 0;
};

WorldSettings LoadWorldSettings(const std::string& path);
