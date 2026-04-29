#pragma once

#include <fstream>
#include <string>
#include <string_view>

class Logger
{
public:
    void initialize(const std::wstring& logDirectory);
    void info(std::string_view message);
    void warn(std::string_view message);
    void error(std::string_view message);

    const std::wstring& logPath() const;

private:
    void write(std::string_view level, std::string_view message);

    std::wstring logPath_;
    std::ofstream stream_;
};
