#define NOMINMAX
#include <Windows.h>

#include "core/Logger.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace
{
std::string timestampForFilename()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
    localtime_s(&localTime, &time);

    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y%m%d-%H%M%S");
    return stream.str();
}

std::string timestampForLine()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
    localtime_s(&localTime, &time);

    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}
}

void Logger::initialize(const std::wstring& logDirectory)
{
    std::filesystem::create_directories(std::filesystem::path(logDirectory));

    const std::string baseName = timestampForFilename();
    for (int suffix = 0; suffix < 1000; ++suffix)
    {
        std::wstring filename(baseName.begin(), baseName.end());
        if (suffix > 0)
        {
            filename += L"-" + std::to_wstring(suffix);
        }
        filename += L".log";

        const std::filesystem::path candidate = std::filesystem::path(logDirectory) / filename;
        if (std::filesystem::exists(candidate))
        {
            continue;
        }

        logPath_ = candidate.wstring();
        break;
    }

    if (logPath_.empty())
    {
        throw std::runtime_error("Failed to allocate a log filename.");
    }

    stream_.open(std::filesystem::path(logPath_), std::ios::out | std::ios::binary);
    if (!stream_)
    {
        throw std::runtime_error("Failed to create log file.");
    }

    info("App started");
}

void Logger::info(std::string_view message)
{
    write("INFO", message);
}

void Logger::warn(std::string_view message)
{
    write("WARN", message);
}

void Logger::error(std::string_view message)
{
    write("ERROR", message);
}

const std::wstring& Logger::logPath() const
{
    return logPath_;
}

void Logger::write(std::string_view level, std::string_view message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_)
    {
        return;
    }

    stream_ << '[' << timestampForLine() << "] [" << level << "] " << message << '\n';
    stream_.flush();
}
