#pragma once

#include <cstdint>
#include <string>

std::wstring formatFixedWidth(double value, int width, int precision);
std::wstring versionString(std::uint32_t version);
std::wstring narrowToWide(const char* text);
std::wstring getCpuBrandString();
std::uint64_t getProcessRamUsageMb();
std::uint64_t getTotalRamMb();
