#define NOMINMAX
#include <Windows.h>
#include <psapi.h>
#include <vulkan/vulkan.h>

#include "core/SystemInfo.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <intrin.h>
#include <sstream>

std::wstring formatFixedWidth(double value, int width, int precision)
{
    std::wostringstream stream;
    stream << std::fixed << std::setprecision(precision) << std::setw(width) << std::setfill(L'0') << value;
    return stream.str();
}

std::wstring versionString(std::uint32_t version)
{
    std::wostringstream stream;
    stream << VK_VERSION_MAJOR(version) << L'.'
           << VK_VERSION_MINOR(version) << L'.'
           << VK_VERSION_PATCH(version);
    return stream.str();
}

std::wstring narrowToWide(const char* text)
{
    if (text == nullptr)
    {
        return L"";
    }

    const int length = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (length <= 1)
    {
        return L"";
    }

    std::wstring result(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, result.data(), length);
    result.pop_back();
    return result;
}

std::wstring getCpuBrandString()
{
    int cpuInfo[4] = {};
    __cpuid(cpuInfo, 0x80000000);
    const unsigned int maxExtendedId = static_cast<unsigned int>(cpuInfo[0]);
    if (maxExtendedId < 0x80000004)
    {
        return L"N/A";
    }

    char brand[49] = {};
    for (unsigned int id = 0; id < 3; ++id)
    {
        int brandInfo[4] = {};
        __cpuid(brandInfo, static_cast<int>(0x80000002 + id));
        std::memcpy(brand + id * 16, brandInfo, sizeof(brandInfo));
    }

    std::string trimmed = brand;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch)
    {
        return !std::isspace(ch);
    }));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch)
    {
        return !std::isspace(ch);
    }).base(), trimmed.end());

    return narrowToWide(trimmed.c_str());
}

std::uint64_t getProcessRamUsageMb()
{
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters)) == 0)
    {
        return 0;
    }

    return static_cast<std::uint64_t>(counters.WorkingSetSize / (1024ull * 1024ull));
}

std::uint64_t getTotalRamMb()
{
    MEMORYSTATUSEX memoryStatus{};
    memoryStatus.dwLength = sizeof(memoryStatus);
    if (GlobalMemoryStatusEx(&memoryStatus) == 0)
    {
        return 0;
    }

    return static_cast<std::uint64_t>(memoryStatus.ullTotalPhys / (1024ull * 1024ull));
}
