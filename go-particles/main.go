package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// Particle represents a single particle in the simulation
type Particle struct {
	X, Y     float64 // Position
	VX, VY   float64 // Velocity
	Mass     float64
	Color    color.RGBA
}

// Simulation holds the state of the particle simulation
type Simulation struct {
	Particles []Particle
	Width     float64
	Height    float64
	DeltaTime float64
}

// NewSimulation creates a new particle simulation
func NewSimulation(width, height float64, numParticles int) *Simulation {
	sim := &Simulation{
		Width:     width,
		Height:    height,
		DeltaTime: 1.0 / 60.0, // 60 FPS
		Particles: make([]Particle, numParticles),
	}

	// Initialize particles with random positions and velocities
	for i := range sim.Particles {
		sim.Particles[i] = Particle{
			X:     rand.Float64() * width,
			Y:     rand.Float64() * height,
			VX:    (rand.Float64() - 0.5) * 100,
			VY:    (rand.Float64() - 0.5) * 100,
			Mass:  1.0,
			Color: color.RGBA{
				R: uint8(rand.Float64() * 255),
				G: uint8(rand.Float64() * 255),
				B: uint8(rand.Float64() * 255),
				A: 255,
			},
		}
	}

	return sim
}

// Update updates the simulation using parallel processing
func (s *Simulation) Update() {
	numWorkers := runtime.NumCPU()
	particlesPerWorker := len(s.Particles) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		start := i * particlesPerWorker
		end := start + particlesPerWorker
		if i == numWorkers-1 {
			end = len(s.Particles)
		}

		go func(start, end int) {
			defer wg.Done()
			s.updateParticlesRange(start, end)
		}(start, end)
	}

	wg.Wait()
}

// updateParticlesRange updates a range of particles
func (s *Simulation) updateParticlesRange(start, end int) {
	for i := start; i < end; i++ {
		s.updateParticle(i)
	}
}

// updateParticle updates a single particle
func (s *Simulation) updateParticle(index int) {
	particle := &s.Particles[index]

	// Apply gravity
	particle.VY -= 50 * s.DeltaTime

	// Update position
	particle.X += particle.VX * s.DeltaTime
	particle.Y += particle.VY * s.DeltaTime

	// Boundary collision with damping
	if particle.X < 0 || particle.X > s.Width {
		particle.VX *= -0.8
		particle.X = math.Max(0, math.Min(s.Width, particle.X))
	}
	if particle.Y < 0 || particle.Y > s.Height {
		particle.VY *= -0.8
		particle.Y = math.Max(0, math.Min(s.Height, particle.Y))
	}

	// Add some air resistance
	particle.VX *= 0.999
	particle.VY *= 0.999
}

// Game represents the Ebiten game
type Game struct {
	simulation *Simulation
	lastTime   time.Time
	frames     int
	lastFPS    time.Time
}

// Update updates the game state
func (g *Game) Update() error {
	now := time.Now()
	g.lastTime = now

	// Update simulation
	g.simulation.Update()

	// Update FPS counter
	g.frames++
	if now.Sub(g.lastFPS).Seconds() >= 1.0 {
		ebiten.SetWindowTitle(fmt.Sprintf("Go Particle Simulation - FPS: %d", g.frames))
		g.frames = 0
		g.lastFPS = now
	}

	return nil
}

// Draw draws the game
func (g *Game) Draw(screen *ebiten.Image) {
	// Clear screen
	screen.Fill(color.RGBA{0, 0, 0, 255})

	// Draw particles
	for _, particle := range g.simulation.Particles {
		// Draw a small circle for each particle
		ebitenutil.DrawRect(screen, particle.X-1, particle.Y-1, 2, 2, particle.Color)
	}
}

// Layout returns the game layout
func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 800, 600
}

func main() {
	// Create simulation with 1000 particles
	sim := NewSimulation(800, 600, 1000)

	game := &Game{
		simulation: sim,
		lastTime:   time.Now(),
		lastFPS:    time.Now(),
	}

	ebiten.SetWindowTitle("Go Particle Simulation - Parallel Processing")
	ebiten.SetWindowSize(800, 600)
	ebiten.SetWindowResizable(true)

	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
