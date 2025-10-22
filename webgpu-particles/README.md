# WebGPU Particle Simulation

This directory contains a WebGPU implementation of a particle simulation for comparing parallel computation approaches.

## Features

- **GPU-accelerated computation**: Uses WebGPU compute shaders for parallel particle updates
- **Real-time rendering**: 1000+ particles with smooth 60 FPS performance
- **Physics simulation**: Gravity, collision detection, and air resistance
- **Modern web technology**: Uses the latest WebGPU API

## How to Run

1. Open `index.html` in a WebGPU-compatible browser (Chrome 113+, Firefox 110+)

## Technical Details

- **Parallel Processing**: Uses WebGPU compute shaders to update all particles in parallel on the GPU
- **Workgroup Size**: 64 particles per workgroup for optimal GPU utilization
- **Memory Layout**: Particles stored in GPU buffers for efficient access
