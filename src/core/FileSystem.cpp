#define NOMINMAX
#include <Windows.h>

#include "core/FileSystem.h"

std::wstring sourcePathWide(const wchar_t* relativePath)
{
    return std::wstring(VULKAN_VOXEL_SOURCE_DIR_WIDE) + relativePath;
}

std::wstring sourcePathWide(const std::wstring& relativePath)
{
    return std::wstring(VULKAN_VOXEL_SOURCE_DIR_WIDE) + relativePath;
}

std::string sourcePath(const char* relativePath)
{
    return std::string(VULKAN_VOXEL_SOURCE_DIR) + relativePath;
}

std::wstring asciiToWide(const std::string& text)
{
    return std::wstring(text.begin(), text.end());
}

bool fileExists(const std::wstring& path)
{
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}
