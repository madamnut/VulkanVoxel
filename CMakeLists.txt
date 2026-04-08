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

add_executable(VulkanVoxel
    VulkanVoxel.cpp
    VulkanVoxel.h
)

target_link_libraries(VulkanVoxel
    PRIVATE
        glfw
        Vulkan::Vulkan
)

if(MSVC)
    target_compile_options(VulkanVoxel PRIVATE /W4 /permissive-)
else()
    target_compile_options(VulkanVoxel PRIVATE -Wall -Wextra -Wpedantic)
endif()
