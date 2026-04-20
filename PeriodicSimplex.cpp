#include "PeriodicSimplex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace PeriodicSimplex {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTau = 2.0 * kPi;
constexpr double kF5 = (2.4494897427831780982 - 1.0) / 5.0;
constexpr double kG5 = (6.0 - 2.4494897427831780982) / 30.0;
constexpr double kAttenuationRadiusSquared = 0.6;
constexpr float kSimplex5Scale = 20.0f;

int FastFloor(double value) {
    const int truncated = static_cast<int>(value);
    return value < static_cast<double>(truncated) ? (truncated - 1) : truncated;
}

std::uint64_t MixBits(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

std::uint64_t Hash5(int seed, int x, int y, int z, int w, int v) {
    std::uint64_t hash = MixBits(static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)) + 0x9e3779b97f4a7c15ull);

    const auto mixCoordinate = [&](int coordinate, std::uint64_t multiplier) {
        const std::uint64_t value = static_cast<std::uint64_t>(static_cast<std::uint32_t>(coordinate));
        hash = MixBits(hash ^ MixBits(value * multiplier));
    };

    mixCoordinate(x, 0x632be59bd9b4e019ull);
    mixCoordinate(y, 0x8cb92ba72f3d8dd7ull);
    mixCoordinate(z, 0x9e3779b185ebca87ull);
    mixCoordinate(w, 0xc2b2ae3d27d4eb4full);
    mixCoordinate(v, 0x165667b19e3779f9ull);
    return hash;
}

float GradientDot(std::uint64_t hash, double x, double y, double z, double w, double v) {
    const int gradientIndex = static_cast<int>(hash % 80ull);
    const int zeroAxis = gradientIndex / 16;
    const int signBits = gradientIndex % 16;

    const double coordinates[5] = {x, y, z, w, v};
    int signBitIndex = 0;
    double result = 0.0;
    for (int axis = 0; axis < 5; ++axis) {
        if (axis == zeroAxis) {
            continue;
        }

        const double sign = (signBits & (1 << signBitIndex)) != 0 ? 1.0 : -1.0;
        result += sign * coordinates[axis];
        ++signBitIndex;
    }

    return static_cast<float>(result);
}

float ComputeCornerContribution(std::uint64_t hash, double x, double y, double z, double w, double v) {
    const double falloff = kAttenuationRadiusSquared - (x * x + y * y + z * z + w * w + v * v);
    if (falloff <= 0.0) {
        return 0.0f;
    }

    const double falloffSquared = falloff * falloff;
    return static_cast<float>(falloffSquared * falloffSquared) * GradientDot(hash, x, y, z, w, v);
}

}  // namespace

float SampleSimplex5D(double x, double y, double z, double w, double v, int seed) {
    const double skew = (x + y + z + w + v) * kF5;
    const int i = FastFloor(x + skew);
    const int j = FastFloor(y + skew);
    const int k = FastFloor(z + skew);
    const int l = FastFloor(w + skew);
    const int m = FastFloor(v + skew);

    const double unskew = static_cast<double>(i + j + k + l + m) * kG5;
    const double cellOriginX = static_cast<double>(i) - unskew;
    const double cellOriginY = static_cast<double>(j) - unskew;
    const double cellOriginZ = static_cast<double>(k) - unskew;
    const double cellOriginW = static_cast<double>(l) - unskew;
    const double cellOriginV = static_cast<double>(m) - unskew;

    const double x0 = x - cellOriginX;
    const double y0 = y - cellOriginY;
    const double z0 = z - cellOriginZ;
    const double w0 = w - cellOriginW;
    const double v0 = v - cellOriginV;

    int rankX = 0;
    int rankY = 0;
    int rankZ = 0;
    int rankW = 0;
    int rankV = 0;

    if (x0 > y0) { ++rankX; } else { ++rankY; }
    if (x0 > z0) { ++rankX; } else { ++rankZ; }
    if (x0 > w0) { ++rankX; } else { ++rankW; }
    if (x0 > v0) { ++rankX; } else { ++rankV; }
    if (y0 > z0) { ++rankY; } else { ++rankZ; }
    if (y0 > w0) { ++rankY; } else { ++rankW; }
    if (y0 > v0) { ++rankY; } else { ++rankV; }
    if (z0 > w0) { ++rankZ; } else { ++rankW; }
    if (z0 > v0) { ++rankZ; } else { ++rankV; }
    if (w0 > v0) { ++rankW; } else { ++rankV; }

    const int i1 = rankX >= 4 ? 1 : 0;
    const int j1 = rankY >= 4 ? 1 : 0;
    const int k1 = rankZ >= 4 ? 1 : 0;
    const int l1 = rankW >= 4 ? 1 : 0;
    const int m1 = rankV >= 4 ? 1 : 0;

    const int i2 = rankX >= 3 ? 1 : 0;
    const int j2 = rankY >= 3 ? 1 : 0;
    const int k2 = rankZ >= 3 ? 1 : 0;
    const int l2 = rankW >= 3 ? 1 : 0;
    const int m2 = rankV >= 3 ? 1 : 0;

    const int i3 = rankX >= 2 ? 1 : 0;
    const int j3 = rankY >= 2 ? 1 : 0;
    const int k3 = rankZ >= 2 ? 1 : 0;
    const int l3 = rankW >= 2 ? 1 : 0;
    const int m3 = rankV >= 2 ? 1 : 0;

    const int i4 = rankX >= 1 ? 1 : 0;
    const int j4 = rankY >= 1 ? 1 : 0;
    const int k4 = rankZ >= 1 ? 1 : 0;
    const int l4 = rankW >= 1 ? 1 : 0;
    const int m4 = rankV >= 1 ? 1 : 0;

    const double x1 = x0 - static_cast<double>(i1) + kG5;
    const double y1 = y0 - static_cast<double>(j1) + kG5;
    const double z1 = z0 - static_cast<double>(k1) + kG5;
    const double w1 = w0 - static_cast<double>(l1) + kG5;
    const double v1 = v0 - static_cast<double>(m1) + kG5;

    const double x2 = x0 - static_cast<double>(i2) + 2.0 * kG5;
    const double y2 = y0 - static_cast<double>(j2) + 2.0 * kG5;
    const double z2 = z0 - static_cast<double>(k2) + 2.0 * kG5;
    const double w2 = w0 - static_cast<double>(l2) + 2.0 * kG5;
    const double v2 = v0 - static_cast<double>(m2) + 2.0 * kG5;

    const double x3 = x0 - static_cast<double>(i3) + 3.0 * kG5;
    const double y3 = y0 - static_cast<double>(j3) + 3.0 * kG5;
    const double z3 = z0 - static_cast<double>(k3) + 3.0 * kG5;
    const double w3 = w0 - static_cast<double>(l3) + 3.0 * kG5;
    const double v3 = v0 - static_cast<double>(m3) + 3.0 * kG5;

    const double x4 = x0 - static_cast<double>(i4) + 4.0 * kG5;
    const double y4 = y0 - static_cast<double>(j4) + 4.0 * kG5;
    const double z4 = z0 - static_cast<double>(k4) + 4.0 * kG5;
    const double w4 = w0 - static_cast<double>(l4) + 4.0 * kG5;
    const double v4 = v0 - static_cast<double>(m4) + 4.0 * kG5;

    const double x5 = x0 - 1.0 + 5.0 * kG5;
    const double y5 = y0 - 1.0 + 5.0 * kG5;
    const double z5 = z0 - 1.0 + 5.0 * kG5;
    const double w5 = w0 - 1.0 + 5.0 * kG5;
    const double v5 = v0 - 1.0 + 5.0 * kG5;

    const float n0 = ComputeCornerContribution(Hash5(seed, i, j, k, l, m), x0, y0, z0, w0, v0);
    const float n1 = ComputeCornerContribution(Hash5(seed, i + i1, j + j1, k + k1, l + l1, m + m1), x1, y1, z1, w1, v1);
    const float n2 = ComputeCornerContribution(Hash5(seed, i + i2, j + j2, k + k2, l + l2, m + m2), x2, y2, z2, w2, v2);
    const float n3 = ComputeCornerContribution(Hash5(seed, i + i3, j + j3, k + k3, l + l3, m + m3), x3, y3, z3, w3, v3);
    const float n4 = ComputeCornerContribution(Hash5(seed, i + i4, j + j4, k + k4, l + l4, m + m4), x4, y4, z4, w4, v4);
    const float n5 = ComputeCornerContribution(Hash5(seed, i + 1, j + 1, k + 1, l + 1, m + 1), x5, y5, z5, w5, v5);

    return (n0 + n1 + n2 + n3 + n4 + n5) * kSimplex5Scale;
}

double WrapToUnit(float value, int period) {
    if (period <= 0) {
        return 0.0;
    }

    double wrapped = std::fmod(static_cast<double>(value), static_cast<double>(period));
    if (wrapped < 0.0) {
        wrapped += static_cast<double>(period);
    }

    return wrapped / static_cast<double>(period);
}

float SampleTileableXZ3DOctave(float worldX, float worldY, float worldZ, const FbmSettings& settings, float frequency, int octaveSeed) {
    const float featureScaleXZ = std::max(settings.featureScaleXZ, 1.0f);
    const float featureScaleY = std::max(settings.featureScaleY, 1.0f);
    const int wrapSizeXZ = std::max(settings.wrapSizeXZ, 1);

    const double angleX = WrapToUnit(worldX, wrapSizeXZ) * kTau;
    const double angleZ = WrapToUnit(worldZ, wrapSizeXZ) * kTau;
    const double torusRadius = (static_cast<double>(wrapSizeXZ) * static_cast<double>(frequency)) /
        (kTau * static_cast<double>(featureScaleXZ));
    const double yScaled = (static_cast<double>(worldY) * static_cast<double>(frequency)) /
        static_cast<double>(featureScaleY);

    return SampleSimplex5D(
        std::cos(angleX) * torusRadius,
        std::sin(angleX) * torusRadius,
        yScaled,
        std::cos(angleZ) * torusRadius,
        std::sin(angleZ) * torusRadius,
        octaveSeed
    );
}

float SampleTileableXZ3DFbm(float worldX, float worldY, float worldZ, const FbmSettings& settings) {
    const int octaveCount = std::max(settings.octaves, 1);

    float sum = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float amplitudeSum = 0.0f;

    for (int octave = 0; octave < octaveCount; ++octave) {
        const int octaveSeed = settings.seed + octave * 1013;
        sum += SampleTileableXZ3DOctave(worldX, worldY, worldZ, settings, frequency, octaveSeed) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= settings.gain;
        frequency *= settings.lacunarity;
    }

    if (amplitudeSum <= 0.0f) {
        return 0.0f;
    }

    return std::clamp(sum / amplitudeSum, -1.0f, 1.0f);
}

}  // namespace PeriodicSimplex
