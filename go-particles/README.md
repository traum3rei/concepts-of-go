# Go Particle Simulation

This directory contains a Go implementation of a particle simulation using CPU parallel processing.

## Features

- **CPU parallel processing**: Uses goroutines and worker pools for parallel computation
- **Real-time rendering**: 1000+ particles with smooth 60 FPS performance
- **Physics simulation**: Gravity, collision detection, and air resistance
- **Cross-platform**: Runs on any platform with Go support

## How to Run

1. Install dependencies:
   ```bash
   go mod tidy
   ```

2. Run the simulation:
   ```bash
   go run main.go
   ```

## Technical Details

- **Parallel Processing**: Uses goroutines to distribute particle updates across CPU cores
- **Worker Pool**: Automatically scales to the number of CPU cores available
- **Rendering**: Uses the Ebiten library for hardware-accelerated graphics

## Performance Comparison

This implementation demonstrates:
- CPU parallel processing with goroutines
- Memory allocation patterns
- Garbage collection impact
- Cross-platform performance characteristics
