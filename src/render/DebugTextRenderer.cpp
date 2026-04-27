#define NOMINMAX
#include <Windows.h>

#include "render/DebugTextRenderer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

std::vector<std::uint8_t> DebugTextRenderer::renderGlyphAtlas()
{
    const int textureWidth = static_cast<int>(kDebugFontAtlasWidth);
    const int textureHeight = static_cast<int>(kDebugFontAtlasHeight);

    BITMAPINFO bitmapInfo{};
    bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfo.bmiHeader.biWidth = textureWidth;
    bitmapInfo.bmiHeader.biHeight = -textureHeight;
    bitmapInfo.bmiHeader.biPlanes = 1;
    bitmapInfo.bmiHeader.biBitCount = 32;
    bitmapInfo.bmiHeader.biCompression = BI_RGB;

    void* bitmapBits = nullptr;
    HDC screenDc = GetDC(nullptr);
    HDC memoryDc = CreateCompatibleDC(screenDc);
    HBITMAP bitmap = CreateDIBSection(screenDc, &bitmapInfo, DIB_RGB_COLORS, &bitmapBits, nullptr, 0);
    ReleaseDC(nullptr, screenDc);

    if (memoryDc == nullptr || bitmap == nullptr || bitmapBits == nullptr)
    {
        if (bitmap != nullptr)
        {
            DeleteObject(bitmap);
        }
        if (memoryDc != nullptr)
        {
            DeleteDC(memoryDc);
        }
        throw std::runtime_error("Failed to create debug glyph atlas bitmap.");
    }

    std::memset(bitmapBits, 0, static_cast<std::size_t>(textureWidth * textureHeight * 4));
    HGDIOBJ oldBitmap = SelectObject(memoryDc, bitmap);
    HFONT font = CreateFontW(
        -42,
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"VCR OSD Mono");
    bool ownsFont = true;
    if (font == nullptr)
    {
        font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
        ownsFont = false;
    }
    HGDIOBJ oldFont = SelectObject(memoryDc, font);

    SetBkMode(memoryDc, TRANSPARENT);
    SetTextColor(memoryDc, RGB(255, 255, 255));
    TEXTMETRICW textMetric{};
    GetTextMetricsW(memoryDc, &textMetric);

    const auto* bgra = static_cast<const std::uint8_t*>(bitmapBits);

    for (int glyphIndex = 0; glyphIndex < kDebugGlyphCount; ++glyphIndex)
    {
        const wchar_t character = static_cast<wchar_t>(kDebugGlyphFirst + glyphIndex);
        const int column = glyphIndex % kDebugGlyphColumns;
        const int row = glyphIndex / kDebugGlyphColumns;
        const int x = column * kDebugGlyphCellWidth;
        const int y = row * kDebugGlyphCellHeight;
        SIZE glyphSize{};
        GetTextExtentPoint32W(memoryDc, &character, 1, &glyphSize);

        TextOutW(memoryDc, x + 2, y + 2, &character, 1);

        DebugGlyph& glyph = glyphs_[static_cast<std::size_t>(glyphIndex)];
        const int glyphWidth = std::min(kDebugGlyphCellWidth, static_cast<int>(glyphSize.cx) + 4);
        const int glyphHeight = std::min(kDebugGlyphCellHeight, static_cast<int>(textMetric.tmHeight) + 4);
        glyph.u0 = static_cast<float>(x) / static_cast<float>(textureWidth);
        glyph.v0 = static_cast<float>(y) / static_cast<float>(textureHeight);
        glyph.u1 = static_cast<float>(x + glyphWidth) / static_cast<float>(textureWidth);
        glyph.v1 = static_cast<float>(y + glyphHeight) / static_cast<float>(textureHeight);
        glyph.advance = static_cast<float>(std::max(1L, static_cast<LONG>(glyphSize.cx)));
        glyph.width = static_cast<float>(glyphWidth);
        glyph.height = static_cast<float>(glyphHeight);
    }

    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(textureWidth * textureHeight * 4));
    for (int i = 0; i < textureWidth * textureHeight; ++i)
    {
        const std::uint8_t b = bgra[i * 4 + 0];
        const std::uint8_t g = bgra[i * 4 + 1];
        const std::uint8_t r = bgra[i * 4 + 2];
        rgba[i * 4 + 0] = 255;
        rgba[i * 4 + 1] = 255;
        rgba[i * 4 + 2] = 255;
        rgba[i * 4 + 3] = std::max({r, g, b});
    }

    SelectObject(memoryDc, oldFont);
    SelectObject(memoryDc, oldBitmap);
    if (ownsFont)
    {
        DeleteObject(font);
    }
    DeleteObject(bitmap);
    DeleteDC(memoryDc);

    return rgba;
}

std::vector<DebugTextVertex> DebugTextRenderer::buildVertices(
    const std::vector<std::wstring>& leftLines,
    const std::vector<std::wstring>& rightLines,
    const std::vector<std::wstring>& bottomLeftLines,
    std::uint32_t viewportWidth,
    std::uint32_t viewportHeight) const
{
    std::vector<DebugTextVertex> vertices;
    vertices.reserve(6000);

    appendTextBlock(vertices, leftLines, 12.0f, 10.0f, false, viewportWidth, viewportHeight);
    appendTextBlock(
        vertices,
        rightLines,
        static_cast<float>(viewportWidth) - 12.0f,
        10.0f,
        true,
        viewportWidth,
        viewportHeight);
    const float bottomLeftY = std::max(
        10.0f,
        static_cast<float>(viewportHeight) -
            10.0f -
            57.0f * static_cast<float>(bottomLeftLines.size()));
    appendTextBlock(vertices, bottomLeftLines, 12.0f, bottomLeftY, false, viewportWidth, viewportHeight);

    if (vertices.size() > kMaxDebugTextVertices)
    {
        vertices.resize(kMaxDebugTextVertices);
    }
    return vertices;
}

void DebugTextRenderer::appendTextBlock(
    std::vector<DebugTextVertex>& vertices,
    const std::vector<std::wstring>& lines,
    float x,
    float y,
    bool alignRight,
    std::uint32_t viewportWidth,
    std::uint32_t viewportHeight) const
{
    for (const std::wstring& line : lines)
    {
        const float lineWidth = measureTextLine(line);
        const float lineX = alignRight ? x - lineWidth : x;

        appendTextLine(vertices, line, lineX - 1.0f, y, {0.0f, 0.0f, 0.0f, 1.0f}, viewportWidth, viewportHeight);
        appendTextLine(vertices, line, lineX + 1.0f, y, {0.0f, 0.0f, 0.0f, 1.0f}, viewportWidth, viewportHeight);
        appendTextLine(vertices, line, lineX, y - 1.0f, {0.0f, 0.0f, 0.0f, 1.0f}, viewportWidth, viewportHeight);
        appendTextLine(vertices, line, lineX, y + 1.0f, {0.0f, 0.0f, 0.0f, 1.0f}, viewportWidth, viewportHeight);
        appendTextLine(vertices, line, lineX, y, {1.0f, 1.0f, 1.0f, 1.0f}, viewportWidth, viewportHeight);

        y += 57.0f;
    }
}

float DebugTextRenderer::measureTextLine(const std::wstring& line) const
{
    float width = 0.0f;
    for (wchar_t character : line)
    {
        width += glyphForCharacter(character).advance;
    }
    return width;
}

const DebugGlyph& DebugTextRenderer::glyphForCharacter(wchar_t character) const
{
    if (character < kDebugGlyphFirst || character > kDebugGlyphLast)
    {
        character = L'?';
    }

    return glyphs_[static_cast<std::size_t>(character - kDebugGlyphFirst)];
}

void DebugTextRenderer::appendTextLine(
    std::vector<DebugTextVertex>& vertices,
    const std::wstring& line,
    float x,
    float y,
    std::array<float, 4> color,
    std::uint32_t viewportWidth,
    std::uint32_t viewportHeight) const
{
    for (wchar_t character : line)
    {
        const DebugGlyph& glyph = glyphForCharacter(character);
        appendGlyphQuad(vertices, x, y, glyph, color, viewportWidth, viewportHeight);
        x += glyph.advance;
    }
}

void DebugTextRenderer::appendGlyphQuad(
    std::vector<DebugTextVertex>& vertices,
    float x,
    float y,
    const DebugGlyph& glyph,
    std::array<float, 4> color,
    std::uint32_t viewportWidth,
    std::uint32_t viewportHeight) const
{
    const float x0 = -1.0f + (2.0f * x / static_cast<float>(viewportWidth));
    const float y0 = 1.0f - (2.0f * y / static_cast<float>(viewportHeight));
    const float x1 = -1.0f + (2.0f * (x + glyph.width) / static_cast<float>(viewportWidth));
    const float y1 = 1.0f - (2.0f * (y + glyph.height) / static_cast<float>(viewportHeight));

    const auto makeVertex = [&](float px, float py, float u, float v) -> DebugTextVertex
    {
        return {{px, py}, {u, v}, {color[0], color[1], color[2], color[3]}};
    };

    vertices.push_back(makeVertex(x0, y0, glyph.u0, glyph.v0));
    vertices.push_back(makeVertex(x1, y0, glyph.u1, glyph.v0));
    vertices.push_back(makeVertex(x1, y1, glyph.u1, glyph.v1));
    vertices.push_back(makeVertex(x0, y0, glyph.u0, glyph.v0));
    vertices.push_back(makeVertex(x1, y1, glyph.u1, glyph.v1));
    vertices.push_back(makeVertex(x0, y1, glyph.u0, glyph.v1));
}
