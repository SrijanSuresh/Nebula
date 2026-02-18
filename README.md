# Titan-Nebula: High-Performance GPU Volumetric Renderer
![1 Million Stars](./assets/milstars.gif)
*Caption: 1 Million particles rendered with volumetric smoothing and orbital tilt.*
![100,000 Stars](./assets/hundk2.gif)
*Caption: 100,000 particles.*
![20,000 Stars](./assets/twentk.gif)
*Caption: 20,000 particles.*

A real-time, 1-million-particle nebula simulation powered by **CUDA** and **OpenGL**. This project simulates differential galactic rotation and volumetric gas dynamics using an optimized heterogeneous computing pipeline.

## Systems Architecture
This project focuses on high-throughput data pipelines and cross-environment stability (WSL2/Linux).

* **Heterogeneous Memory Bridge**: Implemented a manual memory bridge between CUDA and OpenGL to bypass WSL2 interop limitations (`cudaGraphicsGLRegisterBuffer` OS-call failures).
* **Volumetric Simulation**: Utilizes an additive blending model with Gaussian point-smoothing to simulate interstellar gas density rather than discrete points.
* **GPGPU Physics**: Offloaded 1,000,000 independent particle updates to the GPU, utilizing parallel trig math (differential rotation) to achieve real-time performance on an RTX 4060.

## Tech Stack
* **Language**: C++20, CUDA C++
* **Graphics**: OpenGL 3.3 (GLAD/GLFW)
* **Hardware Target**: NVIDIA RTX 40-Series (WSL2 Environment)
* **Algorithms**: Box-Muller Transform, Additive Alpha Blending, Differential Rotation.

## Performance Analysis
* **Throughput**: Successfully maintains stable execution at 1M particles, processing ~12MB of vertex data per frame across the PCIe bus.
* **Optimizations**: Used `cudaMemcpyDeviceToHost` into mapped OpenGL buffers to ensure 100% driver compatibility across Windows/Linux host boundaries.

## Visuals
* **Color Profile**: Deep Golden Orange core transitioning to Royal Purple edges.
* **Perspective**: Tilted 3D field-of-view for depth perception of the galactic plane.

## 🔨 Build Instructions
```bash
make clean
make all
./bin/titan_nebula
