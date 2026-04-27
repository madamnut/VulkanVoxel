# Version 0.0.0.0 Plan

## Goal

Vulkan/C++ 기반 게임 프로젝트의 초기 뼈대를 만든다.

## Scope

- CMake와 Ninja 기반 Windows x64 빌드 구조를 준비한다.
- GLFW를 사용해 Vulkan을 사용할 수 있는 창을 생성한다.
- `assets/` 폴더를 게임 에셋 보관 위치로 사용한다.
- `docs/` 폴더에 버전별 개발 문서 관리 체계를 만든다.
- `config/world.json`의 청크 로딩 반경을 기준으로 플레이어 주변 청크를 생성해 초기 rock 지형을 렌더링한다.
- 초기 월드 높이맵은 저장 없이 월드 좌표 기반 deterministic sin/cos 굴곡으로 생성한다.
- fly 카메라와 debug text를 통해 초기 렌더링 상태를 확인한다.

## Initial Technical Direction

- Language: C++20
- Graphics API: Vulkan
- Windowing: GLFW 3.4
- Build: CMake + Ninja
- Target: Windows x64

## Workflow

- 프로젝트 소유자가 Visual Studio 또는 x64 개발자 터미널에서 직접 빌드한다.
- 자동화 에이전트는 사용자가 명시적으로 요청하지 않는 한 configure/build/run 명령을 실행하지 않는다.
