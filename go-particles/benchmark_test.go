package main

import (
	"fmt"
	"runtime"
	"testing"
)

// BenchmarkUpdate benchmarks the parallel particle update with different particle counts
// Using smaller counts due to O(n²) N-body physics complexity
func BenchmarkUpdate(b *testing.B) {
	particleCounts := []int{1000, 5000, 10000}

	for _, count := range particleCounts {
		b.Run(fmt.Sprintf("particles_%d", count), func(b *testing.B) {
			sim := NewSimulation(800, 600, count)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sim.Update()
			}
		})
	}
}

// BenchmarkUpdateSequential benchmarks sequential (single-threaded) particle update
func BenchmarkUpdateSequential(b *testing.B) {
	particleCounts := []int{1000, 5000, 10000}

	for _, count := range particleCounts {
		b.Run(fmt.Sprintf("particles_%d", count), func(b *testing.B) {
			sim := NewSimulation(800, 600, count)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Sequential update - no goroutines
				sim.updateParticlesRange(0, len(sim.Particles))
			}
		})
	}
}

// BenchmarkUpdateSingleParticle benchmarks updating a single particle
func BenchmarkUpdateSingleParticle(b *testing.B) {
	sim := NewSimulation(800, 600, 1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sim.updateParticle(0)
	}
}

// BenchmarkParallelScaling tests how the simulation scales with different worker counts
func BenchmarkParallelScaling(b *testing.B) {
	numParticles := 10000
	workerCounts := []int{1, 2, 4, 8, 16}

	for _, workers := range workerCounts {
		if workers > runtime.NumCPU() {
			continue // Skip if more workers than CPUs
		}
		b.Run(fmt.Sprintf("workers_%d", workers), func(b *testing.B) {
			sim := NewSimulation(800, 600, numParticles)
			originalNumCPU := runtime.GOMAXPROCS(workers)
			defer runtime.GOMAXPROCS(originalNumCPU)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sim.Update()
			}
		})
	}
}

// BenchmarkMemoryAllocation benchmarks memory allocation patterns
func BenchmarkMemoryAllocation(b *testing.B) {
	b.Run("NewSimulation_1000", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = NewSimulation(800, 600, 1000)
		}
	})

	b.Run("NewSimulation_10000", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = NewSimulation(800, 600, 10000)
		}
	})
}

// BenchmarkCompareParallelVsSequential provides a direct comparison
func BenchmarkCompareParallelVsSequential(b *testing.B) {
	numParticles := 10000
	sim := NewSimulation(800, 600, numParticles)

	b.Run("parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sim.Update()
		}
	})

	b.Run("sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sim.updateParticlesRange(0, len(sim.Particles))
		}
	})
}
