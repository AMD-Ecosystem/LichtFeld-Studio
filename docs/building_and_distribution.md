# Building and Distribution

## Requirements

- CUDA Toolkit 12.8+
- CMake 3.30+
- vcpkg (`VCPKG_ROOT` environment variable set)
- GCC 14+ (Linux) or Visual Studio 2022 v17.10+ (Windows)

On AMD GPUs, ROCm takes the place of the CUDA Toolkit and vcpkg is not used. See [AMD GPUs (ROCm/HIP)](#amd-gpus-rocmhip).

## Linux Prerequisites

On Linux, LichtFeld Studio requires SDL3 to be built with at least one windowing backend (`x11` or `wayland`).
SDL3's vcpkg port can otherwise build "successfully" and produce a binary that fails at startup with `No available video device`.

Debian/Ubuntu packages used by CI:

```bash
sudo apt install \
  git curl unzip cmake gcc-14 g++-14 ccache ninja-build zip tar pkg-config python3 python3-dev \
  libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev \
  libwayland-dev libxkbcommon-dev libegl-dev libdecor-0-dev libibus-1.0-dev libdbus-1-dev \
  libsystemd-dev nasm autoconf autoconf-archive automake libtool
```

The configure step now fails early if neither a usable X11 stack nor a usable Wayland stack is present.
If you intentionally want a headless or experimental build, pass `-DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF`.

## Build Options

### 1. Native Build (Development)

Builds for your GPU only. Fastest compile time.

```bash
cmake -B build
cmake --build build -j 16
./build/LichtFeld-Studio --help

# Example training run
./build/LichtFeld-Studio -d /path/to/data -o /path/to/output
```

### 2. Portable Build (Distribution)

Creates a self-contained package that works on any machine with an NVIDIA driver.

```bash
cmake -B build -DBUILD_PORTABLE=ON
cmake --build build -j 16
cmake --install build --prefix ./dist

./dist/bin/run_lichtfeld.sh --help

# Example training run
./dist/bin/run_lichtfeld.sh -d /path/to/data -o /path/to/output
```

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

`-DUSE_HIP=ON` configures a HIP build for AMD GPUs. It covers the GPU compute libraries only: the tensor library, the training kernels (Adam, schedulers, losses, SSIM, MCMC, densification, bilateral grid, ppisp) and the fastgs, gsplat and edge rasterizer backends, plus a focused GPU test binary named `lfs_compute_tests`. The desktop application (Vulkan viewer, Python plugins, MCP server, USD export, nvImageCodec) is not built on this path and still requires CUDA. The CUDA build is unaffected: with `USE_HIP` set, the top-level `CMakeLists.txt` hands the configure to `cmake/HipCompute.cmake` and returns before any NVIDIA logic runs.

Dependencies are supplied directly rather than through vcpkg:

- ROCm 7.x (tested with 7.2), with `clang++` and `clang` from `$ROCM_PATH/llvm/bin`. `ROCM_PATH` defaults to `/opt/rocm`.
- GLM 1.0.1 or newer, passed as `-DLFS_GLM_INCLUDE_DIR`. Older GLM has no HIP path and its math will not compile in device code.
- Taywee/args, passed as `-DLFS_ARGS_INCLUDE_DIR`.
- spdlog, nlohmann-json, TBB, OpenImageIO and OpenMesh from the distribution packages.
- With `-DBUILD_TESTS=ON`, GoogleTest and a ROCm build of LibTorch, which the tensor tests use as a comparison oracle. Point `Torch_DIR` at it; the `external/libtorch` archive the CUDA build downloads is an NVIDIA build and will not link here.

```bash
cmake -B build-hip -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DLFS_GLM_INCLUDE_DIR=/path/to/glm \
  -DLFS_ARGS_INCLUDE_DIR=/path/to/args \
  -DTorch_DIR=/path/to/rocm/libtorch/share/cmake/Torch
cmake --build build-hip -j 16

# Compute tests, on a single GPU
HIP_VISIBLE_DEVICES=0 ./build-hip/cmake/hip_tests/lfs_compute_tests
```

`CMAKE_HIP_ARCHITECTURES` is optional: CMake detects the architecture of the installed GPU when it is not given. Set it to build for another target, for example `gfx1100` on RDNA 3 or `gfx90a` on CDNA 2. No source change is needed to switch architectures.

Do not pass `-DCMAKE_HIP_COMPILER`. It provokes a second configure pass that does not re-apply `USE_HIP`, and the build then falls back to the CUDA path. HIP is enabled at `project()` time instead.

`lfs_compute_tests` is a plain GoogleTest binary and is not registered with CTest on this path; run it directly, and run it serially, since concurrent processes on one GPU produce spurious failures.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PORTABLE` | OFF | Create self-contained distribution |
| `BUILD_CUDA_PTX_ONLY` | OFF | PTX-only build (auto-enabled by PORTABLE) |
| `BUILD_CUDA_MIN_SM` | 75 | Minimum GPU (75=Turing, 80=Ampere, 89=Ada) |
| `BUILD_TESTS` | OFF | Build test suite |
| `LFS_ENFORCE_LINUX_GUI_BACKENDS` | ON | Linux only. Fail configure if SDL3 would be built without both X11 and Wayland |
| `USE_HIP` | OFF | Build the GPU compute libraries with HIP for AMD GPUs. See [AMD GPUs (ROCm/HIP)](#amd-gpus-rocmhip) |

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
