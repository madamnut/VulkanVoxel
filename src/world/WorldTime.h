#pragma once

#include "core/Math.h"

#include <cmath>
#include <cstdint>

constexpr std::uint64_t kWorldTicksPerMinute = 20;
constexpr std::uint64_t kWorldMinutesPerHour = 60;
constexpr std::uint64_t kWorldHoursPerDay = 24;
constexpr std::uint64_t kWorldTicksPerHour = kWorldTicksPerMinute * kWorldMinutesPerHour;
constexpr std::uint64_t kWorldTicksPerDay = kWorldTicksPerHour * kWorldHoursPerDay;
constexpr std::uint64_t kDefaultWorldTimeTicks = 6 * kWorldTicksPerHour;
constexpr float kWorldTicksPerRealSecond = 20.0f;

struct WorldTimeParts
{
    std::uint64_t day = 0;
    int hour = 0;
    int minute = 0;
    int tick = 0;
};

inline WorldTimeParts splitWorldTime(std::uint64_t totalTicks)
{
    const std::uint64_t day = totalTicks / kWorldTicksPerDay;
    const std::uint64_t ticksOfDay = totalTicks % kWorldTicksPerDay;
    const std::uint64_t totalMinutes = ticksOfDay / kWorldTicksPerMinute;
    return {
        day,
        static_cast<int>(totalMinutes / kWorldMinutesPerHour),
        static_cast<int>(totalMinutes % kWorldMinutesPerHour),
        static_cast<int>(ticksOfDay % kWorldTicksPerMinute),
    };
}

inline Vec3 sunDirectionFromWorldTime(std::uint64_t totalTicks)
{
    const float dayProgress = static_cast<float>(totalTicks % kWorldTicksPerDay) /
        static_cast<float>(kWorldTicksPerDay);
    const float angle = (dayProgress - 0.25f) * 2.0f * kPi;
    return normalize({std::cos(angle), std::sin(angle), 0.0f});
}
