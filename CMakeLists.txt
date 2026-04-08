cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(VulkanVoxel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(glfw-3.4)

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

find_program(
    GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/Bin32"
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK or add glslc to PATH.")
endif()

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(GENERATED_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")
file(MAKE_DIRECTORY "${GENERATED_ASSET_DIR}")

set(CHARACTER_GLB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/assets/Character.glb")
set(CHARACTER_MESH_OUTPUT "${GENERATED_ASSET_DIR}/character_mesh.bin")
set(CHARACTER_TEXTURE_OUTPUT "${GENERATED_ASSET_DIR}/character_texture.bin")

set(SHADER_SOURCES
    shaders/sky.vert
    shaders/sky.frag
    shaders/world.vert
    shaders/world.frag
    shaders/overlay.vert
    shaders/overlay.frag
    shaders/selection.vert
    shaders/selection.frag
)

set(SPIRV_OUTPUTS)
foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME "${SHADER_SOURCE}" NAME)
    set(SPIRV_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT "${SPIRV_OUTPUT}"
        COMMAND "${GLSLC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}" -o "${SPIRV_OUTPUT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}"
        COMMENT "Compile ${SHADER_NAME}"
        VERBATIM
    )

    list(APPEND SPIRV_OUTPUTS "${SPIRV_OUTPUT}")
endforeach()

add_custom_target(VulkanVoxelShaders ALL DEPENDS ${SPIRV_OUTPUTS})

add_custom_command(
    OUTPUT "${CHARACTER_MESH_OUTPUT}" "${CHARACTER_TEXTURE_OUTPUT}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/extract_character_glb.py"
            "${CHARACTER_GLB_SOURCE}"
            "${CHARACTER_MESH_OUTPUT}"
            "${CHARACTER_TEXTURE_OUTPUT}"
    DEPENDS
        "${CHARACTER_GLB_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/extract_character_glb.py"
    COMMENT "Extract Character.glb"
    VERBATIM
)

add_custom_target(
    VulkanVoxelAssets ALL
    DEPENDS "${CHARACTER_MESH_OUTPUT}" "${CHARACTER_TEXTURE_OUTPUT}"
)

add_executable(VulkanVoxel
    MathTypes.h
    VoxelWorld.cpp
    VoxelWorld.h
    VulkanVoxel.h
    VulkanVoxel.cpp
    VulkanVoxelDraw.cpp
    VulkanVoxelResources.cpp
    VulkanVoxelSetup.cpp
)

add_dependencies(VulkanVoxel VulkanVoxelShaders)
add_dependencies(VulkanVoxel VulkanVoxelAssets)

target_compile_definitions(VulkanVoxel
    PRIVATE
        SHADER_DIR="${SHADER_OUTPUT_DIR}"
        ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        CHARACTER_MESH_PATH="${CHARACTER_MESH_OUTPUT}"
        CHARACTER_TEXTURE_PATH="${CHARACTER_TEXTURE_OUTPUT}"
)

target_link_libraries(VulkanVoxel
    PRIVATE
        glfw
        Vulkan::Vulkan
        dxgi
        advapi32
        gdi32
        windowscodecs
        ole32
)

if(MSVC)
    target_compile_options(VulkanVoxel PRIVATE /W4 /permissive-)
else()
    target_compile_options(VulkanVoxel PRIVATE -Wall -Wextra -Wpedantic)
endif()
