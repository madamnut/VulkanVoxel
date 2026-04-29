#pragma once

#include <array>
#include <cstdint>

class SimplexNoise5D
{
public:
    struct Vec5
    {
        float value[5]{};
    };

    static constexpr int kGradientCount = 40;

    explicit SimplexNoise5D(std::uint64_t seed = 0);

    void setSeed(std::uint64_t seed);
    float sample(float x, float y, float z, float w, float v) const;

private:
    static constexpr int kPermutationSize = 256;
    static constexpr int kPermutationMask = kPermutationSize - 1;

    static std::uint64_t nextRandom(std::uint64_t& state);
    static float dot(const Vec5& gradient, const std::array<float, 5>& offset);
    int hash(int i, int j, int k, int l, int m) const;

    std::array<std::uint8_t, kPermutationSize * 2> permutation_{};
};
