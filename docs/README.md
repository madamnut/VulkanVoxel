# VulkanVoxel Documentation Guide

이 폴더는 프로젝트의 버전별 개발 상황과 기획 결정을 누적해서 기록합니다.

## Version Folders

- 시작 버전은 `0.0.0.0`입니다.
- 버전마다 `docs/<version>/` 폴더를 만듭니다.
- 현재 버전에서 생기는 기획, 구현 기록, 변경 내역은 해당 버전 폴더에 계속 누적합니다.
- 다음 버전으로 넘어가기 전에 현재 버전 폴더의 `summary.md`에 총정리를 남깁니다.
- 새 버전이 시작되면 새 폴더를 만들고 같은 방식으로 기록합니다.

## Recommended Files

- `plan.md`: 현재 버전의 목표, 시스템 기획, 해야 할 일
- `log.md`: 날짜별 개발 기록과 의사결정
- `summary.md`: 다음 버전으로 넘어가기 전 최종 정리

## Build Policy

- 빌드 시스템은 CMake와 Ninja를 기준으로 합니다.
- 기본 대상 플랫폼은 Windows x64입니다.
- Debug 빌드는 `windows-x64-debug`, Release 빌드는 `windows-x64-release` CMake preset을 사용합니다.
- 빌드와 실행 확인은 프로젝트 소유자가 직접 수행합니다.
- 자동화 에이전트는 사용자가 명시적으로 요청하지 않는 한 configure/build/run 명령을 실행하지 않습니다.
- Windows에서는 x64 컴파일러 환경이 잡힌 터미널에서 아래 명령을 실행합니다.

```powershell
cmake --preset windows-x64-debug
cmake --build --preset windows-x64-debug
```

- Release 빌드는 `windows-x64-release`로 preset 이름만 바꿔 실행합니다.

## Asset Policy

- 게임에서 사용하는 스프라이트, 텍스처, 폰트, 모델 등은 루트 `assets/` 폴더에 보관합니다.
- 코드에서 참조하는 에셋 경로는 프로젝트 루트를 기준으로 문서화합니다.
- 에셋의 출처, 제작 상태, 교체 예정 여부는 해당 버전의 문서에 기록합니다.

## Config Policy

- 실행 시 조정 가능한 설정은 루트 `config/` 폴더에 보관합니다.
- `config/world.json`의 `chunkLoadRadius`는 플레이어가 속한 청크를 중심으로 로딩할 반경입니다.
- 로딩 청크 수는 `(2n + 1)^2`이며, 설정 변경은 다음 실행부터 적용합니다.
- 현재 `chunkLoadRadius`는 안전을 위해 `0`에서 `64` 사이로 clamp합니다.
- `chunkUploadsPerFrame`은 worker thread가 완성한 청크 메쉬를 프레임당 몇 개까지 GPU upload할지 정합니다.
- 현재 `chunkUploadsPerFrame`은 안전을 위해 `1`에서 `64` 사이로 clamp합니다.
- `chunkBuildThreads`는 청크 CPU 메싱에 사용할 worker thread 개수입니다.
- 현재 `chunkBuildThreads`는 안전을 위해 `1`에서 `16` 사이로 clamp합니다.
- `config/blocks.json`은 블럭 ID, 이름, solid 여부를 정의합니다.
- 블럭 텍스처는 `assets/textures/block/<name>.png`를 기본으로 사용하고, `<name>_top.png`, `<name>_side.png`, `<name>_bottom.png`가 있으면 해당 면에 우선 적용합니다.
- 실행 중 hot reload는 아직 지원하지 않습니다.
