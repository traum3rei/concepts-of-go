package main

import (
	"fmt"
	"testing"
)

// BenchmarkLargeScale benchmarks with large particle counts to compare with GPU
// Warning: These benchmarks are slow due to O(n²) N-body physics
func BenchmarkLargeScale(b *testing.B) {
	particleCounts := []int{10000, 50000, 100000}

	for _, count := range particleCounts {
		b.Run(fmt.Sprintf("parallel_%d", count), func(b *testing.B) {
			sim := NewSimulation(800, 600, count)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sim.Update()
			}
		})
	}
}

// BenchmarkLargeScaleSequential benchmarks sequential update at large scale
func BenchmarkLargeScaleSequential(b *testing.B) {
	// Only test 10K and 50K sequentially - 100K would take too long
	particleCounts := []int{10000, 50000}

	for _, count := range particleCounts {
		b.Run(fmt.Sprintf("sequential_%d", count), func(b *testing.B) {
			sim := NewSimulation(800, 600, count)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sim.updateParticlesRange(0, len(sim.Particles))
			}
		})
	}
}
