class ParticleBenchmark {
    constructor() {
        this.device = null;
        this.results = [];
        this.timestampSupported = false;
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();

        // Check for timestamp query support
        const features = [];
        if (adapter.features.has('timestamp-query')) {
            features.push('timestamp-query');
            this.timestampSupported = true;
        }

        this.device = await adapter.requestDevice({ requiredFeatures: features });

        // Create offscreen canvas for benchmarking
        this.canvas = document.createElement('canvas');
        this.canvas.width = 800;
        this.canvas.height = 600;

        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
        });

    }

    createShaders() {
        this.computeShader = this.device.createShaderModule({
            code: `
                struct Particle {
                    position: vec2<f32>,
                    velocity: vec2<f32>,
                    color: vec4<f32>,
                }

                struct Uniforms {
                    deltaTime: f32,
                    width: f32,
                    height: f32,
                    gravity: f32,
                    _padding: f32,
                }

                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> uniforms: Uniforms;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&particles)) {
                        return;
                    }

                    let particle = particles[index];
                    var force = vec2<f32>(0.0, 0.0);

                    // Gravity (positive Y is down in screen coords)
                    force.y += uniforms.gravity;

                    // Calculate forces from nearby particles
                    let smoothingRadius = 25.0;

                    for (var i = 0u; i < arrayLength(&particles); i++) {
                        if (i == index) {
                            continue;
                        }

                        let other = particles[i];
                        let dx = particle.position.x - other.position.x;
                        let dy = particle.position.y - other.position.y;
                        let distSq = dx * dx + dy * dy;

                        if (distSq < smoothingRadius * smoothingRadius && distSq > 0.01) {
                            let distance = sqrt(distSq);
                            let normalX = dx / distance;
                            let normalY = dy / distance;

                            // Pressure force (repulsion) - smooth falloff
                            let q = 1.0 - distance / smoothingRadius;
                            let pressureForce = 500.0 * q * q;
                            force.x += normalX * pressureForce;
                            force.y += normalY * pressureForce;

                            // Viscosity force - light smoothing
                            let viscosityForce = 0.02 * q;
                            force.x += (other.velocity.x - particle.velocity.x) * viscosityForce;
                            force.y += (other.velocity.y - particle.velocity.y) * viscosityForce;
                        }
                    }

                    // Apply forces to velocity
                    particles[index].velocity += force * uniforms.deltaTime;

                    // Clamp velocity to prevent instability
                    let maxSpeed = 500.0;
                    let speed = length(particles[index].velocity);
                    if (speed > maxSpeed) {
                        particles[index].velocity = (particles[index].velocity / speed) * maxSpeed;
                    }

                    // Update position
                    particles[index].position += particles[index].velocity * uniforms.deltaTime;

                    // Boundary collision with bounce
                    let margin = 5.0;
                    let bounce = 0.5;

                    if (particles[index].position.x < margin) {
                        particles[index].velocity.x = abs(particles[index].velocity.x) * bounce;
                        particles[index].position.x = margin;
                    } else if (particles[index].position.x > uniforms.width - margin) {
                        particles[index].velocity.x = -abs(particles[index].velocity.x) * bounce;
                        particles[index].position.x = uniforms.width - margin;
                    }

                    if (particles[index].position.y < margin) {
                        particles[index].velocity.y = abs(particles[index].velocity.y) * bounce;
                        particles[index].position.y = margin;
                    } else if (particles[index].position.y > uniforms.height - margin) {
                        particles[index].velocity.y = -abs(particles[index].velocity.y) * bounce;
                        particles[index].position.y = uniforms.height - margin;
                    }

                    // Light damping for stability
                    particles[index].velocity *= 0.998;
                }
            `
        });

        this.vertexShader = this.device.createShaderModule({
            code: `
                struct Particle {
                    position: vec2<f32>,
                    velocity: vec2<f32>,
                    color: vec4<f32>,
                }

                struct Uniforms {
                    deltaTime: f32,
                    width: f32,
                    height: f32,
                    gravity: f32,
                    _padding: f32,
                }

                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) color: vec4<f32>,
                }

                @group(0) @binding(0) var<storage, read> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> uniforms: Uniforms;

                @vertex
                fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                    let particleIndex = vertexIndex / 6u;
                    let vertexInParticle = vertexIndex % 6u;

                    let particle = particles[particleIndex];
                    let size = 4.0;

                    let offsets = array<vec2<f32>, 6>(
                        vec2<f32>(-size, -size),
                        vec2<f32>( size, -size),
                        vec2<f32>(-size,  size),
                        vec2<f32>( size, -size),
                        vec2<f32>( size,  size),
                        vec2<f32>(-size,  size)
                    );

                    let offset = offsets[vertexInParticle];
                    let worldPos = particle.position + offset;

                    let clipPos = vec2<f32>(
                        (worldPos.x / uniforms.width)*2.0 - 1.0,
                        1.0 - (worldPos.y / uniforms.height)*2.0
                    );

                    var output: VertexOutput;
                    output.position = vec4<f32>(clipPos, 0.0, 1.0);
                    output.color = particle.color;
                    return output;
                }
            `
        });

        this.fragmentShader = this.device.createShaderModule({
            code: `
                @fragment
                fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                    return color;
                }
            `
        });
    }

    createBuffers(numParticles) {
        const particleSize = 8 * 4;
        const particleBuffer = this.device.createBuffer({
            size: numParticles * particleSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        const particleData = new Float32Array(numParticles * 8);
        const centerX = 400;
        const centerY = 300;
        const blobRadius = 80;

        for (let i = 0; i < numParticles; i++) {
            const baseIndex = i * 8;
            const angle = (i / numParticles) * 2 * Math.PI;
            const radius = Math.random() * blobRadius;
            particleData[baseIndex + 0] = centerX + Math.cos(angle) * radius;
            particleData[baseIndex + 1] = centerY + Math.sin(angle) * radius;
            particleData[baseIndex + 2] = 0;
            particleData[baseIndex + 3] = 0;
            particleData[baseIndex + 4] = 0.1 + Math.random() * 0.3;
            particleData[baseIndex + 5] = 0.3 + Math.random() * 0.4;
            particleData[baseIndex + 6] = 0.7 + Math.random() * 0.3;
            particleData[baseIndex + 7] = 1.0;
        }

        this.device.queue.writeBuffer(particleBuffer, 0, particleData);

        const uniformBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        return { particleBuffer, uniformBuffer };
    }

    createPipelines(particleBuffer, uniformBuffer, numParticles) {
        const computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.computeShader,
                entryPoint: 'main',
            },
        });

        const renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.vertexShader,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: this.fragmentShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        const computeBindGroup = this.device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: uniformBuffer } },
            ],
        });

        const renderBindGroup = this.device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: uniformBuffer } },
            ],
        });

        return { computePipeline, renderPipeline, computeBindGroup, renderBindGroup };
    }

    async runBenchmark(numParticles, iterations = 100) {
        this.createShaders();
        const { particleBuffer, uniformBuffer } = this.createBuffers(numParticles);
        const { computePipeline, renderPipeline, computeBindGroup, renderBindGroup } =
            this.createPipelines(particleBuffer, uniformBuffer, numParticles);

        const uniformData = new Float32Array([1/60, 800.0, 600.0, 200.0, 0.0, 0.0, 0.0, 0.0]);
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        for (let i = 0; i < 10; i++) {
            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));
            computePass.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }
        await this.device.queue.onSubmittedWorkDone();

        // Benchmark compute shader
        const computeTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();

            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));
            computePass.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            const end = performance.now();
            computeTimes.push(end - start);
        }

        // Benchmark render
        const renderTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();

            const commandEncoder = this.device.createCommandEncoder();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, renderBindGroup);
            renderPass.draw(numParticles * 6);
            renderPass.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            const end = performance.now();
            renderTimes.push(end - start);
        }

        // Benchmark full frame (compute + render)
        const frameTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();

            const commandEncoder = this.device.createCommandEncoder();

            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, computeBindGroup);
            computePass.dispatchWorkgroups(Math.ceil(numParticles / 64));
            computePass.end();

            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, renderBindGroup);
            renderPass.draw(numParticles * 6);
            renderPass.end();

            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            const end = performance.now();
            frameTimes.push(end - start);
        }

        // Cleanup
        particleBuffer.destroy();
        uniformBuffer.destroy();

        return {
            particles: numParticles,
            iterations,
            compute: this.calculateStats(computeTimes),
            render: this.calculateStats(renderTimes),
            frame: this.calculateStats(frameTimes),
        };
    }

    calculateStats(times) {
        const sorted = [...times].sort((a, b) => a - b);
        const sum = times.reduce((a, b) => a + b, 0);
        const mean = sum / times.length;
        const variance = times.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / times.length;
        const stdDev = Math.sqrt(variance);

        return {
            mean: mean.toFixed(3),
            min: sorted[0].toFixed(3),
            max: sorted[sorted.length - 1].toFixed(3),
            median: sorted[Math.floor(sorted.length / 2)].toFixed(3),
            stdDev: stdDev.toFixed(3),
            p95: sorted[Math.floor(sorted.length * 0.95)].toFixed(3),
            p99: sorted[Math.floor(sorted.length * 0.99)].toFixed(3),
        };
    }

    async runAllBenchmarks(particleCounts = [1000, 5000, 10000]) {

        const results = [];

        for (const count of particleCounts) {
            console.log(`\nBenchmarking ${count} particles...`);
            const result = await this.runBenchmark(count);
            results.push(result);

            console.log(`  Compute: ${result.compute.mean}ms (±${result.compute.stdDev}ms)`);
            console.log(`  Render:  ${result.render.mean}ms (±${result.render.stdDev}ms)`);
            console.log(`  Frame:   ${result.frame.mean}ms (±${result.frame.stdDev}ms)`);
            console.log(`  Max FPS: ${(1000 / parseFloat(result.frame.mean)).toFixed(1)}`);
        }

        this.results = results;
        return results;
    }

    printResults() {
        console.log('\n' + '='.repeat(60));
        console.log('BENCHMARK RESULTS SUMMARY');
        console.log('='.repeat(60));

        console.log('\n| Particles | Compute (ms) | Render (ms) | Frame (ms) | Max FPS |');
        console.log('|-----------|--------------|-------------|------------|---------|');

        for (const r of this.results) {
            const maxFps = (1000 / parseFloat(r.frame.mean)).toFixed(1);
            console.log(`| ${r.particles.toString().padStart(9)} | ${r.compute.mean.padStart(12)} | ${r.render.mean.padStart(11)} | ${r.frame.mean.padStart(10)} | ${maxFps.padStart(7)} |`);
        }

        console.log('\nNote: Times are in milliseconds');
        console.log('Compute = particle physics simulation on GPU');
        console.log('Render = drawing particles to screen');
        console.log('Frame = total time per frame (compute + render)');
    }

    getResultsAsJSON() {
        return JSON.stringify(this.results, null, 2);
    }
}

async function runBenchmarks() {
    const outputDiv = document.getElementById('output');
    const log = (msg) => {
        console.log(msg);
        if (outputDiv) {
            outputDiv.textContent += msg + '\n';
        }
    };

    try {
        const benchmark = new ParticleBenchmark();
        await benchmark.init();

        log('WebGPU Particle Simulation Benchmark');
        log('=' .repeat(50));
        log(`GPU: ${(await navigator.gpu.requestAdapter()).info?.device || 'Unknown'}`);
        log('');

        const results = await benchmark.runAllBenchmarks([1000, 10000, 50000, 100000]);
        benchmark.printResults();

        window.webgpuBenchmarkResults = results;

        return results;
    } catch (error) {
        log(`Error: ${error.message}`);
        console.error(error);
    }
}

// Export for use as module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ParticleBenchmark, runBenchmarks };
}
