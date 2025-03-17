# CUDA GPU Camera Publisher for Unreal Engine

## Overview

This project implements a CUDA-based GPU plugin integrated with Unreal Engine to capture, process, and publish camera data using the Data Distribution Service (DDS) middleware. It includes two main camera publishers: one for a front-facing camera (`AFront_camera_publisher`) and one for a rear-facing camera (`ARear_camera_publisher`), along with a reusable `UGPGPUComponent` for CUDA kernel management and execution.

The system leverages CUDA for GPU-accelerated image processing (e.g., converting RGBA to RGB) and uses double-buffering to handle frame data efficiently. The processed frames are published over DDS topics for real-time communication, suitable for applications like autonomous vehicles or robotics simulations.

## Features

- **CUDA Integration**: Manages CUDA contexts, streams, and kernel execution within Unreal Engine.
- **Camera Capture**: Captures frames from Unreal Engine's `USceneCaptureComponent2D` into render targets.
- **Image Processing**: Converts RGBA frames to RGB using a CUDA kernel (`RgbaToRgbKernel`).
- **Double Buffering**: Uses a thread-safe double-buffer system (`FFrontBuffer`/`FRearBuffer`) for frame handling.
- **DDS Publishing**: Publishes processed frames to DDS topics (`rt/FrontCameraImageTopic` and `rt/RearCameraImageTopic`) using Cyclone DDS.
- **Threading**: Runs camera processing in dedicated threads (`FrontCameraThread`/`RearCameraThread`) for performance.
- **Error Handling**: Includes cleanup mechanisms for CUDA, DDS, and render targets on failure or shutdown.

## Components

1. **`UGPGPUComponent`**:
   - Initializes and manages CUDA contexts, streams, and kernels.
   - Loads precompiled PTX modules or compiles CUDA source code dynamically using NVRTC.
   - Supports CUDA graph capture for optimized kernel execution.

2. **`AFront_camera_publisher` / `ARear_camera_publisher`**:
   - Unreal Engine actors that capture camera frames, process them using CUDA, and publish them via DDS.
   - Configurable resolution (default: 960x540) and field of view (default: 120°).
   - Uses double render targets for continuous frame capture.

3. **`FFrontBuffer` / `FRearBuffer`**:
   - Thread-safe double-buffering classes for managing frame data between capture and publishing.

## Dependencies

- **CUDA**: Requires CUDA Toolkit (Driver API and NVRTC) for GPU operations.
- **Cyclone DDS**: Used for DDS communication (configuration via `Cyclonedds.xml`).
- **CUDAPlugin**: Assumes a plugin structure with precompiled PTX files (e.g., `matSumKernel.ptx`).

## Setup

1. **Install Dependencies**:
   - Ensure CUDA Toolkit is installed and configured on your system.
   - Include Cyclone DDS in your project (e.g., via a plugin or submodule. I have a separate Cyclonedds plugin in another repo).

2. **Project Configuration**:
   - Place the code in your Unreal Engine project’s plugin or module directory.
   - Ensure `CUDAPlugin/Source/GPGPUPlugin/Public/matSumKernel.ptx` exists, or provide a `UKernelContainer` with CUDA source code.

3. **Run**:
   - Add `AFront_camera_publisher` and/or `ARear_camera_publisher` to your Unreal Engine level.
   - The actors will automatically initialize and start publishing camera data on play.

## Usage

- **Configuration**: Adjust `ResolutionX`, `ResolutionY`, and `FieldOfView` in the actor constructors as needed.
- **DDS Topics**: Subscribe to `rt/FrontCameraImageTopic` or `rt/RearCameraImageTopic` to receive image data.
- **Kernel Customization**: Modify the CUDA kernel in `matSumKernel.ptx` or provide a custom `UKernelContainer`.

## Limitations

- Requires a CUDA-capable GPU.
- Assumes a specific directory structure for PTX files and DDS configuration.
- No real-time tick logic implemented in `Tick()`; processing occurs in separate threads.

## License

This project is provided as-is for educational and development purposes. Refer to your Unreal Engine and CUDA licensing terms for commercial use.

---
Generated on: March 17, 2025
