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

type Particle struct {
	X, Y     float64 // Position
	VX, VY   float64 // Velocity
	Mass     float64
	Color    color.RGBA
}

type Simulation struct {
	Particles []Particle
	Width     float64
	Height    float64
	DeltaTime float64
}

func NewSimulation(width, height float64, numParticles int) *Simulation {
	sim := &Simulation{
		Width:     width,
		Height:    height,
		DeltaTime: 1.0 / 60.0, // 60 FPS
		Particles: make([]Particle, numParticles),
	}

	for i := range sim.Particles {
		sim.Particles[i] = Particle{
			X:     50 + rand.Float64()*700,        
			Y:     50 + rand.Float64()*250,       
			VX:    (rand.Float64() - 0.5) * 50,   
			VY:    rand.Float64() * 20,            
			Mass:  1.0,
			Color: color.RGBA{
				R: uint8(25 + rand.Float64()*75),   
				G: uint8(75 + rand.Float64()*100),
				B: uint8(180 + rand.Float64()*75),
				A: 255,
			},
		}
	}

	return sim
}

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

func (s *Simulation) updateParticlesRange(start, end int) {
	for i := start; i < end; i++ {
		s.updateParticle(i)
	}
}

// updateParticle updates a single particle with N-body physics
func (s *Simulation) updateParticle(index int) {
	particle := &s.Particles[index]

	// Calculate forces
	forceX := 0.0
	forceY := 0.0

	// Apply gravity
	gravity := 200.0
	forceY += gravity

	// Calculate forces from nearby particles (N-body interaction)
	smoothingRadius := 25.0

	for i := range s.Particles {
		if i == index {
			continue
		}

		other := &s.Particles[i]
		dx := particle.X - other.X
		dy := particle.Y - other.Y
		distSq := dx*dx + dy*dy

		if distSq < smoothingRadius*smoothingRadius && distSq > 0.01 {
			distance := math.Sqrt(distSq)
			normalX := dx / distance
			normalY := dy / distance

			// Pressure force (repulsion) 
			q := 1.0 - distance/smoothingRadius
			pressureForce := 500.0 * q * q
			forceX += normalX * pressureForce
			forceY += normalY * pressureForce

			// Viscosity force - light smoothing
			viscosityForce := 0.02 * q
			forceX += (other.VX - particle.VX) * viscosityForce
			forceY += (other.VY - particle.VY) * viscosityForce
		}
	}

	// Apply forces to velocity
	particle.VX += forceX * s.DeltaTime
	particle.VY += forceY * s.DeltaTime

	// Clamp velocity to prevent instability
	maxSpeed := 500.0
	speed := math.Sqrt(particle.VX*particle.VX + particle.VY*particle.VY)
	if speed > maxSpeed {
		particle.VX = (particle.VX / speed) * maxSpeed
		particle.VY = (particle.VY / speed) * maxSpeed
	}

	// Update position
	particle.X += particle.VX * s.DeltaTime
	particle.Y += particle.VY * s.DeltaTime

	// Boundary collision with bounce
	margin := 5.0
	bounce := 0.5

	if particle.X < margin {
		particle.VX = math.Abs(particle.VX) * bounce
		particle.X = margin
	} else if particle.X > s.Width-margin {
		particle.VX = -math.Abs(particle.VX) * bounce
		particle.X = s.Width - margin
	}

	if particle.Y < margin {
		particle.VY = math.Abs(particle.VY) * bounce
		particle.Y = margin
	} else if particle.Y > s.Height-margin {
		particle.VY = -math.Abs(particle.VY) * bounce
		particle.Y = s.Height - margin
	}

	// Light damping for stability
	particle.VX *= 0.998
	particle.VY *= 0.998
}

type Game struct {
	simulation *Simulation
	lastTime   time.Time
	frames     int
	lastFPS    time.Time
}

func (g *Game) Update() error {
	now := time.Now()
	g.lastTime = now

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

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{0, 0, 0, 255})

	for _, particle := range g.simulation.Particles {
		ebitenutil.DrawRect(screen, particle.X-1, particle.Y-1, 2, 2, particle.Color)
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 800, 600
}

func main() {
	sim := NewSimulation(800, 600, 3000)

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
