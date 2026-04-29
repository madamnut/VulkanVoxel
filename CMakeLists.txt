cmake_minimum_required(VERSION 3.24)

project(VulkanVoxel
    VERSION 0.0.0.0
    DESCRIPTION "Vulkan voxel game"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED COMPONENTS glslc)

if(DEFINED Vulkan_GLSLC_EXECUTABLE AND EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
    set(GLSLC_EXECUTABLE "${Vulkan_GLSLC_EXECUTABLE}")
else()
    find_program(GLSLC_EXECUTABLE
        NAMES glslc glslc.exe
        HINTS "$ENV{VULKAN_SDK}/Bin"
        REQUIRED
    )
endif()

file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" VULKAN_VOXEL_SOURCE_DIR)
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}" VULKAN_VOXEL_BINARY_DIR)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw-3.4)

set(SHADER_SOURCES
    shaders/sky.vert
    shaders/sky.frag
    shaders/block.vert
    shaders/block.frag
    shaders/player.vert
    shaders/player.frag
    shaders/selection.vert
    shaders/selection.frag
    shaders/debug_text.vert
    shaders/debug_text.frag
)

set(SHADER_BINARIES)
foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME "${SHADER_SOURCE}" NAME)
    set(SHADER_BINARY "${PROJECT_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT "${SHADER_BINARY}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${PROJECT_BINARY_DIR}/shaders"
        COMMAND "${GLSLC_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/${SHADER_SOURCE}" -o "${SHADER_BINARY}"
        DEPENDS "${PROJECT_SOURCE_DIR}/${SHADER_SOURCE}"
        VERBATIM
    )
    list(APPEND SHADER_BINARIES "${SHADER_BINARY}")
endforeach()

add_custom_target(VulkanVoxelShaders DEPENDS ${SHADER_BINARIES})

add_executable(VulkanVoxel
    src/assets/ImageLoader.cpp
    src/core/FileSystem.cpp
    src/core/GameConfig.cpp
    src/core/Logger.cpp
    src/core/Math.cpp
    src/core/SystemInfo.cpp
    src/player/PlayerController.cpp
    src/render/CrosshairRenderer.cpp
    src/render/DebugTextRenderer.cpp
    src/render/PlayerModel.cpp
    src/render/VulkanDescriptors.cpp
    src/render/VulkanHelpers.cpp
    src/render/VulkanPipelineBuilder.cpp
    src/render/VulkanResourceContext.cpp
    src/render/VulkanSwapchain.cpp
    src/render/TextureManager.cpp
    src/world/BlockRegistry.cpp
    src/world/BlockRaycast.cpp
    src/world/ChunkMesher.cpp
    src/world/ChunkStreamingManager.cpp
    src/world/SimplexNoise.cpp
    src/world/WorldSave.cpp
    src/world/WorldGenerator.cpp
    src/main.cpp
    third_party/lz4/lz4.c
)
add_dependencies(VulkanVoxel VulkanVoxelShaders)

target_compile_definitions(VulkanVoxel PRIVATE
    VULKAN_VOXEL_VERSION="${PROJECT_VERSION}"
    VULKAN_VOXEL_SOURCE_DIR="${VULKAN_VOXEL_SOURCE_DIR}"
    VULKAN_VOXEL_SOURCE_DIR_WIDE=L"${VULKAN_VOXEL_SOURCE_DIR}"
    VULKAN_VOXEL_BINARY_DIR="${VULKAN_VOXEL_BINARY_DIR}"
)

target_compile_options(VulkanVoxel PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

target_link_libraries(VulkanVoxel PRIVATE
    Vulkan::Vulkan
    glfw
    gdi32
    ole32
    psapi
    windowscodecs
)

target_include_directories(VulkanVoxel PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/third_party/lz4"
)

set_target_properties(VulkanVoxel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)
