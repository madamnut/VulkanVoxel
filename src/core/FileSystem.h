#pragma once

#include <string>

std::wstring sourcePathWide(const wchar_t* relativePath);
std::wstring sourcePathWide(const std::wstring& relativePath);
std::string sourcePath(const char* relativePath);
std::wstring asciiToWide(const std::string& text);
bool fileExists(const std::wstring& path);
