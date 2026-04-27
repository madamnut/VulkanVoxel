#define NOMINMAX
#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include "assets/ImageLoader.h"

#include <cstddef>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

ImagePixels loadPngWithWic(const std::wstring& path)
{
    ComPtr<IWICImagingFactory> factory;
    HRESULT result = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory));
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to create WIC imaging factory.");
    }

    ComPtr<IWICBitmapDecoder> decoder;
    result = factory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &decoder);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to open PNG texture.");
    }

    ComPtr<IWICBitmapFrameDecode> frame;
    result = decoder->GetFrame(0, &frame);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to decode PNG frame.");
    }

    ComPtr<IWICFormatConverter> converter;
    result = factory->CreateFormatConverter(&converter);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to create WIC format converter.");
    }

    result = converter->Initialize(
        frame.Get(),
        GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to convert PNG to RGBA.");
    }

    ImagePixels image{};
    result = converter->GetSize(&image.width, &image.height);
    if (FAILED(result) || image.width == 0 || image.height == 0)
    {
        throw std::runtime_error("PNG texture has an invalid size.");
    }

    const std::uint32_t stride = image.width * 4;
    image.rgba.resize(static_cast<std::size_t>(stride) * image.height);
    result = converter->CopyPixels(nullptr, stride, static_cast<UINT>(image.rgba.size()), image.rgba.data());
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to copy PNG pixels.");
    }

    return image;
}
