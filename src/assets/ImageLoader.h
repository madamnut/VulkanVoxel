#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ImagePixels
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgba;
};

ImagePixels loadPngWithWic(const std::wstring& path);
