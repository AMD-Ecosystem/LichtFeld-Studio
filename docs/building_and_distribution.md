# Building and Distribution

## Requirements

- CUDA Toolkit 12.8+
- CMake 3.30+
- vcpkg (`VCPKG_ROOT` environment variable set)
- GCC 14+ (Linux) or Visual Studio 2022 v17.10+ (Windows)

On AMD GPUs, ROCm takes the place of the CUDA Toolkit. See [AMD GPUs (ROCm/HIP)](#amd-gpus-rocmhip).

Windows builds require the **C++ Clang Compiler for Windows** Visual Studio
Installer individual component in addition to the regular C++ desktop workload.
It provides `clang-cl`, which is used only to build libplacebo (an unconditional
dependency of the standard build) with a Microsoft-compatible ABI. The port
locates `clang-cl.exe` on `PATH` or under `VSINSTALLDIR`/`VCINSTALLDIR` (for
example from the x64 Native Tools Command Prompt). **MSBuild support for LLVM
(clang-cl) toolset** is optional and only needed if you want the LLVM toolset
selectable in Visual Studio IDE projects; the vcpkg port itself does not use
MSBuild. The rest of LichtFeld Studio continues to use the configured Visual
Studio/MSVC toolchain.

## Contributor Setup

Install the repository's pre-commit hook after cloning:

```bash
cp tools/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook applies `clang-format` to staged C, C++, and CUDA source files outside
`external/` before each commit.

## Linux Prerequisites

On Linux, LichtFeld Studio requires SDL3 to be built with at least one windowing backend (`x11` or `wayland`).
SDL3's vcpkg port can otherwise build "successfully" and produce a binary that fails at startup with `No available video device`.

Debian/Ubuntu packages used by CI:

```bash
sudo apt install \
  git curl unzip cmake gcc-14 g++-14 ccache ninja-build zip tar pkg-config python3 python3-dev \
  libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev \
  libwayland-dev libxkbcommon-dev libegl-dev libdecor-0-dev libibus-1.0-dev libdbus-1-dev \
  libsystemd-dev libgtk-3-dev nasm autoconf autoconf-archive automake libtool
```

`libgtk-3-dev` is required because `nativefiledialog-extended` is built from source on Linux
against its GTK backend rather than taken from vcpkg; see `cmake/SetupNativeFileDialog.cmake`.
Without it configure fails with `pkg-config could not locate `gtk+-3.0``.

The configure step now fails early if neither a usable X11 stack nor a usable Wayland stack is present.
If you intentionally want a headless or experimental build, pass `-DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF`.

## Build Options

### 1. Native Build (Development)

Builds for your GPU only. Fastest compile time.

```bash
cmake -B build
cmake --build build -j 16
./build/LichtFeld-Studio --help

# Example training run (writes /path/to/output/project.licht; add --headless and --export ply for a CLI-only splat export)
./build/LichtFeld-Studio -d /path/to/data -o /path/to/output
```

#### Release-only dependencies (optional)

Native x64 Windows and Linux builds can use these presets to build only the
Release variants of vcpkg dependencies, including host tools:

| Platform | Release application | Optimized application with debug information |
| --- | --- | --- |
| Windows x64 | `windows-release` | `windows-relwithdebinfo` |
| Linux x64 | `linux-release` | `linux-relwithdebinfo` |

For example, from an initialized Windows x64 MSVC/CUDA development shell:

```sh
cmake --preset windows-release
cmake --build --preset windows-release
```

Each preset uses its own build directory. Release and RelWithDebInfo on the
same platform share the dependency recipe and can reuse compatible binary
cache entries; the first Release-only install may rebuild packages. The
standard `build` and `debug` presets remain available, with tests opt-in.

See the [developer build guide](docs/development/build.md#release-only-dependency-profiles)
for all commands, cache behavior and dependency-symbol coverage. To keep an
existing `cmake --build build ...` command, follow the
[existing Ninja directory migration](docs/development/build.md#keeping-an-existing-ninja-build-directory),
including `-B build` and `-DBUILD_TESTS=ON` when building `lichtfeld_tests`.
The [test prerequisites](#tests) still apply.

### 2. Portable Build (Distribution)

Creates a self-contained package that works on any machine with an NVIDIA driver.

```bash
cmake -B build -DBUILD_PORTABLE=ON
cmake --build build -j 16
cmake --install build --prefix ./dist

./dist/bin/run_lichtfeld.sh --help

# Example training run (writes /path/to/output/project.licht)
./dist/bin/run_lichtfeld.sh -d /path/to/data -o /path/to/output
```

## Tests

The tensor comparison tests validate the built-in tensor library against LibTorch as an
oracle, so a LibTorch SDK is required to configure the test build. It is not needed for the
application itself, which is LibTorch-free.

Download the LibTorch C++ SDK matching your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/) (select **LibTorch** as the package
and **C++/Java** as the language) and unpack it so `TorchConfig.cmake` resolves:

| Platform | Expected location |
|---|---|
| Linux | `external/libtorch/` |
| Windows (Release) | `external/release/libtorch/` |
| Windows (Debug) | `external/debug/libtorch/` |

```bash
# Linux, from the repository root
curl -L -o libtorch.zip "<libtorch-download-url>"
unzip -q libtorch.zip -d external/
```

Configuring without it fails at `find_package(Torch REQUIRED)`. Some LibTorch builds link
CUDA libraries they do not ship; if the test binaries then fail to start with a missing
shared library, install the named library or use a build that bundles its CUDA dependencies.

Tests are a separate opt-in build:

```bash
cmake -S . -B build/tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON
cmake --build build/tests \
  --target lichtfeld_tests tensor_hardening_tests \
  -j6
```

Run the CTest label that matches the subsystem and cost of the change:

| Label | Use it for |
|---|---|
| `fast` | Default developer loop: core unit tests, fast Python tests, focused GPU contracts, and some real-data loader checks |
| `slow` | Dataset loaders, training, interop, and other multi-second integration paths |
| `nightly` | Stress, large-tensor, real-data, and optional-codec coverage |
| `hardening` | Isolated tensor hardening and crash-prone oracle tests |
| `discovery` | Discovery fuzzing and parameter sweeps |
| `gpu` | All registered tests that require GPU execution |

For example:

```bash
ctest --test-dir build/tests --output-on-failure -L fast
ctest --test-dir build/tests --output-on-failure -L slow
ctest --test-dir build/tests --output-on-failure -L nightly
```

### Localization contracts

The localization contracts are a small, headless validation suite. They validate
locale key and placeholder parity, one-key-per-line JSON formatting, literal
translation-key references, RML directives, count-sensitive plural-form rules,
the hardcoded-UI-text audit, and localized cached UI state. They do not build
or execute the GUI, LibTorch, or CUDA test targets. The contributor guide
documents the locale conventions and language-specific grammar policy.

Register the contracts in an existing build directory:

```bash
cmake -S . -B build -DBUILD_LOCALIZATION_TESTS=ON
cmake --build build --target test_localization_contracts
```

The custom target runs the single CTest entry named `LocalizationContracts`.
It can also be invoked directly after configuration:

```bash
ctest --test-dir build/tests --output-on-failure -R LocalizationContracts
```

The repository intentionally ignores `data/`, but several fast, slow, nightly,
and GPU tests read real files from `data/bicycle` or `data/garden`. Populate
those paths with compatible real datasets before running the affected tiers.
Synthetic placeholder input is not a substitute: loader, training, image-codec,
and checkpoint tests assert the expected images, masks, COLMAP files, or point
cloud exist on disk.

## What's the Difference?

| | Native Build | Portable Build |
|---|---|---|
| **Output** | `build/LichtFeld-Studio` (66 MB) | `dist/` folder (518 MB) |
| **Target needs CUDA** | Yes | No |
| **Target needs vcpkg** | Yes | No |
| **Self-contained** | No | Yes |
| **Use case** | Development | End-user distribution |

## Distribution Contents

```
dist/
├── bin/
│   ├── LichtFeld-Studio
│   └── run_lichtfeld.sh    # Use this to launch
├── lib/                    # Bundled CUDA & runtime libs
├── share/LichtFeld-Studio/ # Shaders, icons, fonts
└── LICENSE                 # GPL-3.0
```

## AMD GPUs (ROCm/HIP)

`-DUSE_HIP=ON` builds LichtFeld Studio for AMD GPUs with ROCm/HIP in place of CUDA: the application with its Vulkan viewer, the Python module, the MCP server and `lichtfeld_tests`, with dependencies from vcpkg as in the CUDA build. The CUDA build is unaffected.

Requirements on Linux, besides the [Linux prerequisites](#linux-prerequisites):

- ROCm with its Clang and the rocJPEG and rocDecode libraries (tested with ROCm 10.1, which ships HIP 7.16 and rocDecode 1.10). The build looks for ROCm under `ROCM_PATH`, which defaults to `/opt/rocm`. On Linux it stops at configure if rocDecode is missing. On Windows, rocDecode is part of ROCm from ROCm 10.2 (rocDecode 1.12); with an older ROCm such as 10.1 the build configures with a warning and decodes all video on the CPU.
- GCC 14 for C and C++, as in the CUDA build; the ROCm Clang compiles the HIP sources.
- With `-DBUILD_TESTS=ON`, a ROCm build of LibTorch, passed as `Torch_DIR`. The `external/libtorch` archive the CUDA build downloads is an NVIDIA build and will not link here.

```bash
cmake -B build-hip -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_HIP_COMPILER="${ROCM_PATH:-/opt/rocm}/lib/llvm/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DTorch_DIR=/path/to/rocm/libtorch/share/cmake/Torch
cmake --build build-hip -j 16
./build-hip/LichtFeld-Studio --help
```

On a system ROCm install the HIP compiler is at `${ROCM_PATH}/llvm/bin/clang++`. Keep `CC` and `CXX` set to GCC 14 in the environment while configuring: vcpkg builds the dependencies with them.

What differs from the CUDA build:

- JPEG images decode on the GPU with rocJPEG. ROCm has no JPEG encoder or JPEG 2000 codec, so JPEG encoding, the lossless JPEG 2000 image caches, and JPEGs the hardware decoder does not accept (smaller than 64x64, progressive) are handled on the CPU with libjpeg and OpenJPH.
- On Linux, HIP cannot import Vulkan semaphores, so the viewer orders HIP and Vulkan work on the host around each shared image instead of on the GPU.
- Video decodes on the GPU with rocDecode (H.264, HEVC, VP9 and AV1, as far as the GPU supports them). Streams it cannot decode on the GPU, such as 4:4:4 chroma or MPEG-4 Part 2, decode with FFmpeg's software decoders, and the log names the reason (on a Windows build without rocDecode, "rocDecode not available in this ROCm SDK on Windows"). The first time a process decodes a codec on a GPU, rocDecode decodes a two-frame reference clip and the frames are compared with FFmpeg's software decode, which takes 10 to 50 ms; some GPU and video driver combinations decode without an error but return wrong pixels, and on those the codec decodes on the CPU with "rocDecode produced wrong frames for <codec> on this GPU and driver" in the log. FFmpeg decodes AV1 on the CPU with libdav1d, which the ROCm build adds to FFmpeg's vcpkg features through the `rocm` manifest feature in `vcpkg.json`; the CUDA build decodes AV1 on NVDEC and does not select that feature. ROCm has no video encoder, so video export encodes H.264 with FFmpeg's software encoder.
- With ROCm 10.1.0 from TheRock, the VA-API driver the SDK bundles (`radeonsi_drv_video.so` under `rocm_sysdeps`) returns wrong frames for videos smaller than 720p on some GPUs, seen on MI200 (gfx90a); the check above catches this and those codecs decode on the CPU. On 10.1.0, setting `LIBVA_DRIVERS_PATH` to the distribution's Mesa VA driver directory (on Ubuntu `/usr/lib/x86_64-linux-gnu/dri`, from `mesa-va-drivers`; verified with Mesa 25.2.8) restores GPU decode. Mesa commit fc8ba4378f37 fixes the driver, and SDKs built from TheRock commit 1288c16f or later bundle a Mesa that contains it. From TheRock commit b04778ebc on, the SDK loads its bundled driver ahead of `LIBVA_DRIVERS_PATH`, so the override has no effect there.

### Compute-only build

`-DUSE_HIP=ON -DLFS_HIP_COMPUTE_ONLY=ON` builds only the GPU compute libraries: the tensor library, the training kernels (Adam, schedulers, losses, SSIM, MCMC, densification, bilateral grid, ppisp) and the fastgs and gsplat rasterizer backends, plus a focused GPU test binary named `lfs_compute_tests`. With this option the top-level `CMakeLists.txt` hands the configure to `cmake/HipCompute.cmake` and returns before the application is configured.

Dependencies are supplied directly rather than through vcpkg:

- ROCm 7.x (tested with 7.2), with `clang++` and `clang` from the ROCm LLVM directory. The build looks for ROCm under `ROCM_PATH`, which defaults to `/opt/rocm`; set it if ROCm is installed elsewhere.
- GLM 1.0.1 or newer, passed as `-DLFS_GLM_INCLUDE_DIR`. Older GLM has no HIP path and its math will not compile in device code.
- Taywee/args, passed as `-DLFS_ARGS_INCLUDE_DIR`.
- nlohmann-json, TBB, xxHash, libjpeg, libpng, libtiff, libwebp and OpenMesh from the distribution packages (on Ubuntu, `libjpeg-dev libpng-dev libtiff-dev libwebp-dev libxxhash-dev libopenmesh-dev`). xxHash is used header-only, so only `xxhash.h` is required.
- OpenEXR 3.3 or newer. The `libopenexr-dev` package in Ubuntu 24.04 is 3.1 and too old, so build OpenEXR from source there and pass `-DOpenEXR_DIR=<prefix>/lib/cmake/OpenEXR` or add its install prefix to `CMAKE_PREFIX_PATH`.
- spdlog 1.15 built with its bundled fmt (`-DSPDLOG_FMT_EXTERNAL=OFF`, the default). The `libspdlog-dev` package in Ubuntu 24.04 (spdlog 1.12 on an external fmt 9) does not compile with the ROCm Clang. Point `spdlog_DIR` at the install or add its prefix to `CMAKE_PREFIX_PATH`.
- With `-DBUILD_TESTS=ON`, GoogleTest and a ROCm build of LibTorch, which the tensor tests use as a comparison oracle. Both are found by config package, so point `GTest_DIR` and `Torch_DIR` at them or list their install prefixes in `CMAKE_PREFIX_PATH`. The `external/libtorch` archive the CUDA build downloads is an NVIDIA build and will not link here.

```bash
cmake -B build-hip -G Ninja \
  -DUSE_HIP=ON \
  -DLFS_HIP_COMPUTE_ONLY=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_CXX_COMPILER="${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++" \
  -DCMAKE_C_COMPILER="${ROCM_PATH:-/opt/rocm}/llvm/bin/clang" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DLFS_GLM_INCLUDE_DIR=/path/to/glm \
  -DLFS_ARGS_INCLUDE_DIR=/path/to/args \
  -DGTest_DIR=/path/to/gtest/lib/cmake/GTest \
  -DTorch_DIR=/path/to/rocm/libtorch/share/cmake/Torch
cmake --build build-hip -j 16

# Compute tests, on a single GPU
HIP_VISIBLE_DEVICES=0 ./build-hip/cmake/hip_tests/lfs_compute_tests
```

`CMAKE_HIP_ARCHITECTURES` is optional: CMake detects the architecture of the installed GPU when it is not given. Set it to build for another target, for example `gfx1100` on RDNA 3 or `gfx90a` on CDNA 2. No source change is needed to switch architectures.

The commands above are the Linux recipe. `CMAKE_HIP_COMPILER` does not have to be set when ROCm's `bin` directory is on `PATH`: CMake asks `hipconfig` where the ROCm Clang lives, and HIP is enabled at `project()` time so no reconfigure is needed. Otherwise name the compiler yourself with `-DCMAKE_HIP_COMPILER` or the `HIPCXX` environment variable; setting `ROCM_PATH` alone does not steer that search.

The full application build has not been verified on Windows yet; the compute-only build has. On Windows, configure from a Visual Studio developer prompt so the MSVC libraries and linker are found. C and C++ use `clang-cl` and HIP uses `clang++`, both from the ROCm LLVM directory; HIP is enabled after C and C++, which lets the two front ends coexist. LibTorch comes from a ROCm PyTorch install, whose `LoadHIP.cmake` requires `PYTORCH_ROCM_ARCH` to name the target. The image codec libraries and OpenMesh are not needed on Windows. spdlog, nlohmann-json, TBB, GoogleTest and the `xxhash.h` header are found as on Linux, but Windows has no system location for them: list the install prefixes that hold them in `CMAKE_PREFIX_PATH`. If `clang++` cannot find the device libraries, set `HIP_DEVICE_LIB_PATH` to `%ROCM_PATH%\lib\llvm\amdgcn\bitcode`.

```bat
set PYTORCH_ROCM_ARCH=gfx1151
cmake -B build-hip -G Ninja ^
  -DUSE_HIP=ON ^
  -DLFS_HIP_COMPUTE_ONLY=ON ^
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 ^
  -DCMAKE_C_COMPILER=%ROCM_PATH%\lib\llvm\bin\clang-cl.exe ^
  -DCMAKE_CXX_COMPILER=%ROCM_PATH%\lib\llvm\bin\clang-cl.exe ^
  -DCMAKE_HIP_COMPILER=%ROCM_PATH%\lib\llvm\bin\clang++.exe ^
  -DROCM_PATH=%ROCM_PATH% ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_TESTS=ON ^
  "-DCMAKE_PREFIX_PATH=C:\path\to\deps;C:\path\to\more\deps" ^
  -DLFS_GLM_INCLUDE_DIR=C:\path\to\glm ^
  -DLFS_ARGS_INCLUDE_DIR=C:\path\to\args ^
  -DTorch_DIR=C:\path\to\torch\share\cmake\Torch
cmake --build build-hip --target lfs_compute_tests
build-hip\cmake\hip_tests\lfs_compute_tests.exe
```

The build copies the SDK's `amdhip64` DLL next to `lfs_compute_tests.exe`, because the copy the display driver installs in System32 would otherwise be loaded first. The remaining DLLs come from `PATH`: put the build directory, `torch\lib` and the ROCm `bin` directories on it, with the GPU-specific ROCm libraries ahead of the SDK's own copies, or rocBLAS cannot find the kernels for the GPU.

`lfs_compute_tests` is a plain GoogleTest binary and is not registered with CTest on this path, so run it directly. It executes its tests sequentially in a single process; run one instance at a time.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PORTABLE` | OFF | Create self-contained distribution |
| `BUILD_CUDA_PTX_ONLY` | OFF | PTX-only build (auto-enabled by PORTABLE) |
| `BUILD_CUDA_MIN_SM` | 75 | Minimum GPU (75=Turing, 80=Ampere, 89=Ada) |
| `BUILD_TESTS` | OFF | Build test suite |
| `BUILD_LOCALIZATION_TESTS` | OFF | Register headless localization contract tests |
| `LFS_ENFORCE_LINUX_GUI_BACKENDS` | ON | Linux only. Fail configure if SDL3 would be built without both X11 and Wayland |
| `LFS_CUDA_COMPILER_CACHE` | *(empty)* | Compiler cache for CUDA only. Empty follows the auto-detected launcher; `OFF` disables CUDA caching; or name/path of a launcher such as `ccache`. Needed where nvcc cannot be wrapped by sccache |
| `USE_HIP` | OFF | Build the GPU compute libraries with HIP for AMD GPUs. See [AMD GPUs (ROCm/HIP)](#amd-gpus-rocmhip) |

## Preprocess Model Downloads

The `preprocess` subcommand downloads the default MoGe-2 ONNX model on first
use when `--model` is not provided, then converts it once to a sibling `.lfw`
weight file for the in-tree native runtime. The cached model and every
downloaded temporary file are SHA-256 verified on Windows and Linux before
use. A hash mismatch deletes the untrusted temporary file, rejects the
cached model, and exits with an error. Use `preprocess --download-only` to
preload and verify the cache, or `--no-download` to require an already verified
cache entry.

## Troubleshooting

**"CUDA driver version is insufficient"** - Update NVIDIA driver.

**"no kernel image is available"** - GPU is older than `BUILD_CUDA_MIN_SM`. Rebuild with lower value.

**Missing libraries on target** - Use `run_lichtfeld.sh` (Linux) or ensure DLLs are with .exe (Windows).

**"SDL3 was found, but the resolved SDL build does not expose an X11 or Wayland video backend"** - A stale vcpkg SDL3 artifact is being reused. Remove SDL3 from the local vcpkg install/build cache, then reconfigure with binary-cache bypass enabled:

```bash
lfs="$(pwd)"
triplet="${VCPKG_TARGET_TRIPLET:-x64-linux}"
cd "$VCPKG_ROOT"
./vcpkg remove "sdl3:$triplet" --recurse \
  --x-install-root="$lfs/build/vcpkg_installed" \
  --x-packages-root="$VCPKG_ROOT/packages" \
  --x-buildtrees-root="$VCPKG_ROOT/buildtrees"

cd "$lfs"
VCPKG_BINARY_SOURCES='clear;default,write' cmake -B build -G Ninja --fresh
cmake --build build -j"$(nproc)"
```
