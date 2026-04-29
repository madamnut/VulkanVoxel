#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr int kDebugGlyphFirst = 32;
constexpr int kDebugGlyphLast = 126;
constexpr int kDebugGlyphCount = kDebugGlyphLast - kDebugGlyphFirst + 1;
constexpr int kDebugGlyphColumns = 16;
constexpr int kDebugGlyphCellWidth = 64;
constexpr int kDebugGlyphCellHeight = 64;
constexpr int kDebugFontPixelHeight = 21;
constexpr float kDebugTextLineHeight = 28.5f;
constexpr std::uint32_t kDebugFontAtlasWidth = kDebugGlyphColumns * kDebugGlyphCellWidth;
constexpr std::uint32_t kDebugFontAtlasHeight =
    ((kDebugGlyphCount + kDebugGlyphColumns - 1) / kDebugGlyphColumns) * kDebugGlyphCellHeight;
constexpr std::size_t kMaxDebugTextVertices = 48000;

struct DebugGlyph
{
    float u0 = 0.0f;
    float v0 = 0.0f;
    float u1 = 0.0f;
    float v1 = 0.0f;
    float advance = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

struct DebugTextVertex
{
    float position[2];
    float uv[2];
    float color[4];
};

class DebugTextRenderer
{
public:
    std::vector<std::uint8_t> renderGlyphAtlas();

    std::vector<DebugTextVertex> buildVertices(
        const std::vector<std::wstring>& leftLines,
        const std::vector<std::wstring>& rightLines,
        const std::vector<std::wstring>& bottomLeftLines,
        std::uint32_t viewportWidth,
        std::uint32_t viewportHeight) const;

private:
    void appendTextBlock(
        std::vector<DebugTextVertex>& vertices,
        const std::vector<std::wstring>& lines,
        float x,
        float y,
        bool alignRight,
        std::uint32_t viewportWidth,
        std::uint32_t viewportHeight) const;

    float measureTextLine(const std::wstring& line) const;
    const DebugGlyph& glyphForCharacter(wchar_t character) const;

    void appendTextLine(
        std::vector<DebugTextVertex>& vertices,
        const std::wstring& line,
        float x,
        float y,
        std::array<float, 4> color,
        std::uint32_t viewportWidth,
        std::uint32_t viewportHeight) const;

    void appendGlyphQuad(
        std::vector<DebugTextVertex>& vertices,
        float x,
        float y,
        const DebugGlyph& glyph,
        std::array<float, 4> color,
        std::uint32_t viewportWidth,
        std::uint32_t viewportHeight) const;

    std::array<DebugGlyph, kDebugGlyphCount> glyphs_{};
};
