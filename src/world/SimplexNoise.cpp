#include "world/SimplexNoise.h"

#include "core/Math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace
{
constexpr std::array<SimplexNoise5D::Vec5, SimplexNoise5D::kGradientCount> makeGradients()
{
    std::array<SimplexNoise5D::Vec5, SimplexNoise5D::kGradientCount> gradients{};
    int index = 0;
    for (int a = 0; a < 5; ++a)
    {
        for (int b = a + 1; b < 5; ++b)
        {
            for (int signA = -1; signA <= 1; signA += 2)
            {
                for (int signB = -1; signB <= 1; signB += 2)
                {
                    SimplexNoise5D::Vec5 gradient{};
                    gradient.value[a] = static_cast<float>(signA);
                    gradient.value[b] = static_cast<float>(signB);
                    gradients[static_cast<std::size_t>(index)] = gradient;
                    ++index;
                }
            }
        }
    }
    return gradients;
}

constexpr std::array<SimplexNoise5D::Vec5, SimplexNoise5D::kGradientCount> kGradients5D = makeGradients();

constexpr std::array<SimplexNoise4D::Vec4, SimplexNoise4D::kGradientCount> makeGradients4D()
{
    std::array<SimplexNoise4D::Vec4, SimplexNoise4D::kGradientCount> gradients{};
    int index = 0;
    for (int zeroAxis = 0; zeroAxis < 4; ++zeroAxis)
    {
        for (int signA = -1; signA <= 1; signA += 2)
        {
            for (int signB = -1; signB <= 1; signB += 2)
            {
                for (int signC = -1; signC <= 1; signC += 2)
                {
                    SimplexNoise4D::Vec4 gradient{};
                    int component = 0;
                    for (int axis = 0; axis < 4; ++axis)
                    {
                        if (axis == zeroAxis)
                        {
                            continue;
                        }
                        const int sign = component == 0 ? signA : (component == 1 ? signB : signC);
                        gradient.value[axis] = static_cast<float>(sign);
                        ++component;
                    }
                    gradients[static_cast<std::size_t>(index)] = gradient;
                    ++index;
                }
            }
        }
    }
    return gradients;
}

constexpr std::array<SimplexNoise4D::Vec4, SimplexNoise4D::kGradientCount> kGradients4D = makeGradients4D();

int fastFloor(float value)
{
    const int integer = static_cast<int>(value);
    return value < static_cast<float>(integer) ? integer - 1 : integer;
}
}

SimplexNoise5D::SimplexNoise5D(std::uint64_t seed)
{
    setSeed(seed);
}

void SimplexNoise5D::setSeed(std::uint64_t seed)
{
    std::array<std::uint8_t, kPermutationSize> values{};
    std::iota(values.begin(), values.end(), static_cast<std::uint8_t>(0));

    std::uint64_t state = seed == 0 ? 0x9e3779b97f4a7c15ull : seed;
    for (int i = kPermutationSize - 1; i > 0; --i)
    {
        const std::uint64_t random = nextRandom(state);
        const int swapIndex = static_cast<int>(random % static_cast<std::uint64_t>(i + 1));
        std::swap(values[static_cast<std::size_t>(i)], values[static_cast<std::size_t>(swapIndex)]);
    }

    for (int i = 0; i < kPermutationSize * 2; ++i)
    {
        permutation_[static_cast<std::size_t>(i)] =
            values[static_cast<std::size_t>(i & kPermutationMask)];
    }
}

float SimplexNoise5D::sample(float x, float y, float z, float w, float v) const
{
    constexpr int kDimension = 5;
    constexpr float skewFactor = (2.449489742783178f - 1.0f) / 5.0f;
    constexpr float unskewFactor = (1.0f - 1.0f / 2.449489742783178f) / 5.0f;
    constexpr float contributionRadius = 0.6f;
    constexpr float outputScale = 32.0f;

    const std::array<float, kDimension> input = {x, y, z, w, v};
    const float skew = (x + y + z + w + v) * skewFactor;
    std::array<int, kDimension> cell = {
        fastFloor(x + skew),
        fastFloor(y + skew),
        fastFloor(z + skew),
        fastFloor(w + skew),
        fastFloor(v + skew),
    };

    const float unskew = static_cast<float>(cell[0] + cell[1] + cell[2] + cell[3] + cell[4]) * unskewFactor;
    std::array<float, kDimension> base{};
    std::array<float, kDimension> offset0{};
    for (int axis = 0; axis < kDimension; ++axis)
    {
        base[static_cast<std::size_t>(axis)] = static_cast<float>(cell[static_cast<std::size_t>(axis)]) - unskew;
        offset0[static_cast<std::size_t>(axis)] = input[static_cast<std::size_t>(axis)] - base[static_cast<std::size_t>(axis)];
    }

    std::array<int, kDimension> rank{};
    for (int a = 0; a < kDimension; ++a)
    {
        for (int b = a + 1; b < kDimension; ++b)
        {
            if (offset0[static_cast<std::size_t>(a)] > offset0[static_cast<std::size_t>(b)])
            {
                ++rank[static_cast<std::size_t>(a)];
            }
            else
            {
                ++rank[static_cast<std::size_t>(b)];
            }
        }
    }

    float result = 0.0f;
    for (int corner = 0; corner <= kDimension; ++corner)
    {
        std::array<int, kDimension> step{};
        if (corner > 0)
        {
            const int threshold = kDimension - corner;
            for (int axis = 0; axis < kDimension; ++axis)
            {
                step[static_cast<std::size_t>(axis)] =
                    rank[static_cast<std::size_t>(axis)] >= threshold ? 1 : 0;
            }
        }

        std::array<float, kDimension> offset{};
        float distanceSq = 0.0f;
        for (int axis = 0; axis < kDimension; ++axis)
        {
            offset[static_cast<std::size_t>(axis)] =
                offset0[static_cast<std::size_t>(axis)] -
                static_cast<float>(step[static_cast<std::size_t>(axis)]) +
                static_cast<float>(corner) * unskewFactor;
            distanceSq += offset[static_cast<std::size_t>(axis)] * offset[static_cast<std::size_t>(axis)];
        }

        float attenuation = contributionRadius - distanceSq;
        if (attenuation <= 0.0f)
        {
            continue;
        }

        attenuation *= attenuation;
        const int gradientIndex = hash(
            cell[0] + step[0],
            cell[1] + step[1],
            cell[2] + step[2],
            cell[3] + step[3],
            cell[4] + step[4]) % kGradientCount;
        result += attenuation * attenuation *
            dot(kGradients5D[static_cast<std::size_t>(gradientIndex)], offset);
    }

    return std::clamp(result * outputScale, -1.0f, 1.0f);
}

std::uint64_t SimplexNoise5D::nextRandom(std::uint64_t& state)
{
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

float SimplexNoise5D::dot(const Vec5& gradient, const std::array<float, 5>& offset)
{
    return gradient.value[0] * offset[0] +
           gradient.value[1] * offset[1] +
           gradient.value[2] * offset[2] +
           gradient.value[3] * offset[3] +
           gradient.value[4] * offset[4];
}

int SimplexNoise5D::hash(int i, int j, int k, int l, int m) const
{
    const auto index = [](int value)
    {
        return static_cast<std::size_t>(value & kPermutationMask);
    };

    const int h4 = permutation_[index(m)];
    const int h3 = permutation_[index(l + h4)];
    const int h2 = permutation_[index(k + h3)];
    const int h1 = permutation_[index(j + h2)];
    return permutation_[index(i + h1)];
}

SimplexNoise4D::SimplexNoise4D(std::uint64_t seed)
{
    setSeed(seed);
}

void SimplexNoise4D::setSeed(std::uint64_t seed)
{
    std::array<std::uint8_t, kPermutationSize> values{};
    std::iota(values.begin(), values.end(), static_cast<std::uint8_t>(0));

    std::uint64_t state = seed == 0 ? 0xd1b54a32d192ed03ull : seed;
    for (int i = kPermutationSize - 1; i > 0; --i)
    {
        const std::uint64_t random = nextRandom(state);
        const int swapIndex = static_cast<int>(random % static_cast<std::uint64_t>(i + 1));
        std::swap(values[static_cast<std::size_t>(i)], values[static_cast<std::size_t>(swapIndex)]);
    }

    for (int i = 0; i < kPermutationSize * 2; ++i)
    {
        permutation_[static_cast<std::size_t>(i)] =
            values[static_cast<std::size_t>(i & kPermutationMask)];
    }
}

float SimplexNoise4D::sample(float x, float y, float z, float w) const
{
    constexpr int kDimension = 4;
    constexpr float skewFactor = (2.23606797749979f - 1.0f) / 4.0f;
    constexpr float unskewFactor = (1.0f - 1.0f / 2.23606797749979f) / 4.0f;
    constexpr float contributionRadius = 0.6f;
    constexpr float outputScale = 27.0f;

    const std::array<float, kDimension> input = {x, y, z, w};
    const float skew = (x + y + z + w) * skewFactor;
    std::array<int, kDimension> cell = {
        fastFloor(x + skew),
        fastFloor(y + skew),
        fastFloor(z + skew),
        fastFloor(w + skew),
    };

    const float unskew = static_cast<float>(cell[0] + cell[1] + cell[2] + cell[3]) * unskewFactor;
    std::array<float, kDimension> base{};
    std::array<float, kDimension> offset0{};
    for (int axis = 0; axis < kDimension; ++axis)
    {
        base[static_cast<std::size_t>(axis)] = static_cast<float>(cell[static_cast<std::size_t>(axis)]) - unskew;
        offset0[static_cast<std::size_t>(axis)] = input[static_cast<std::size_t>(axis)] - base[static_cast<std::size_t>(axis)];
    }

    std::array<int, kDimension> rank{};
    for (int a = 0; a < kDimension; ++a)
    {
        for (int b = a + 1; b < kDimension; ++b)
        {
            if (offset0[static_cast<std::size_t>(a)] > offset0[static_cast<std::size_t>(b)])
            {
                ++rank[static_cast<std::size_t>(a)];
            }
            else
            {
                ++rank[static_cast<std::size_t>(b)];
            }
        }
    }

    float result = 0.0f;
    for (int corner = 0; corner <= kDimension; ++corner)
    {
        std::array<int, kDimension> step{};
        if (corner > 0)
        {
            const int threshold = kDimension - corner;
            for (int axis = 0; axis < kDimension; ++axis)
            {
                step[static_cast<std::size_t>(axis)] =
                    rank[static_cast<std::size_t>(axis)] >= threshold ? 1 : 0;
            }
        }

        std::array<float, kDimension> offset{};
        float distanceSq = 0.0f;
        for (int axis = 0; axis < kDimension; ++axis)
        {
            offset[static_cast<std::size_t>(axis)] =
                offset0[static_cast<std::size_t>(axis)] -
                static_cast<float>(step[static_cast<std::size_t>(axis)]) +
                static_cast<float>(corner) * unskewFactor;
            distanceSq += offset[static_cast<std::size_t>(axis)] * offset[static_cast<std::size_t>(axis)];
        }

        float attenuation = contributionRadius - distanceSq;
        if (attenuation <= 0.0f)
        {
            continue;
        }

        attenuation *= attenuation;
        const int gradientIndex = hash(
            cell[0] + step[0],
            cell[1] + step[1],
            cell[2] + step[2],
            cell[3] + step[3]) % kGradientCount;
        result += attenuation * attenuation *
            dot(kGradients4D[static_cast<std::size_t>(gradientIndex)], offset);
    }

    return std::clamp(result * outputScale, -1.0f, 1.0f);
}

std::uint64_t SimplexNoise4D::nextRandom(std::uint64_t& state)
{
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

float SimplexNoise4D::dot(const Vec4& gradient, const std::array<float, 4>& offset)
{
    return gradient.value[0] * offset[0] +
           gradient.value[1] * offset[1] +
           gradient.value[2] * offset[2] +
           gradient.value[3] * offset[3];
}

int SimplexNoise4D::hash(int i, int j, int k, int l) const
{
    const auto index = [](int value)
    {
        return static_cast<std::size_t>(value & kPermutationMask);
    };

    const int h3 = permutation_[index(l)];
    const int h2 = permutation_[index(k + h3)];
    const int h1 = permutation_[index(j + h2)];
    return permutation_[index(i + h1)];
}
