class ParticleSimulation {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.bindGroup = null;
        this.numParticles = 3000;
        this.lastTime = 0;
        this.frameCount = 0;
        this.fpsElement = document.getElementById('fps');
        this.particlesElement = document.getElementById('particles');
        
        this.init();
    }

    async init() {
        try {
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported');
            }

            const adapter = await navigator.gpu.requestAdapter();
            this.device = await adapter.requestDevice();

            this.context = this.canvas.getContext('webgpu');
            const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({
                device: this.device,
                format: canvasFormat,
            });

            await this.setupShaders();
            await this.setupBuffers();
            await this.setupPipeline();
            
            this.animate();
        } catch (error) {
            console.error('Failed to initialize WebGPU:', error);
            document.body.innerHTML = '<div style="color: red; text-align: center; margin-top: 50px;">WebGPU not supported in this browser</div>';
        }
    }

    async setupShaders() {
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

                            // Pressure force 
                            let q = 1.0 - distance / smoothingRadius;
                            let pressureForce = 500.0 * q * q;
                            force.x += normalX * pressureForce;
                            force.y += normalY * pressureForce;

                            // Viscosity 
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
                    let particleIndex = vertexIndex / 6u; // 6 vertices per particle (2 triangles)
                    let vertexInParticle = vertexIndex % 6u;
                    
                    let particle = particles[particleIndex];
                    let size = 2.5; // Particle size
                    
                    // Create two triangles to form a quad for each particle
                    let offsets = array<vec2<f32>, 6>(
                        vec2<f32>(-size, -size), // 0: bottom-left
                        vec2<f32>( size, -size), // 1: bottom-right  
                        vec2<f32>(-size,  size), // 2: top-left
                        vec2<f32>( size, -size), // 3: bottom-right
                        vec2<f32>( size,  size), // 4: top-right
                        vec2<f32>(-size,  size)  // 5: top-left
                    );
                    
                    let offset = offsets[vertexInParticle];
                    let worldPos = particle.position + offset;
                    
                    // Convert to clip space (0,0 to 800,600 -> -1,1 to 1,-1)
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

    async setupBuffers() {
        
        const particleSize = 8 * 4; 
        this.particleBuffer = this.device.createBuffer({
            size: this.numParticles * particleSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        const particleData = new Float32Array(this.numParticles * 8);

        for (let i = 0; i < this.numParticles; i++) {
            const baseIndex = i * 8;

            particleData[baseIndex + 0] = 50 + Math.random() * 700; 
            particleData[baseIndex + 1] = 50 + Math.random() * 250; 
            particleData[baseIndex + 2] = (Math.random() - 0.5) * 50; 
            particleData[baseIndex + 3] = Math.random() * 20;

            particleData[baseIndex + 4] = 0.1 + Math.random() * 0.3; // r
            particleData[baseIndex + 5] = 0.3 + Math.random() * 0.4; // g
            particleData[baseIndex + 6] = 0.7 + Math.random() * 0.3; // b
            particleData[baseIndex + 7] = 1.0; // a
        }

        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);

        
        this.uniformBuffer = this.device.createBuffer({
            size: 32, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    async setupPipeline() {
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.computeShader,
                entryPoint: 'main',
            },
        });

        this.renderPipeline = this.device.createRenderPipeline({
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

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.particleBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                },
            ],
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.particleBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                },
            ],
        });
    }

    updateUniforms(deltaTime) {
        const uniformData = new Float32Array([
            deltaTime,
            800.0, // width
            600.0, // height
            200.0, // gravity 
            0.0,   
            0.0,   
            0.0,   
            0.0,   
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    animate(currentTime = 0) {
        const deltaTime = Math.min((currentTime - this.lastTime) / 1000, 1/30);
        this.lastTime = currentTime;

        this.updateUniforms(deltaTime);

        const commandEncoder = this.device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
        computePass.end();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(this.numParticles * 6); 
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        this.frameCount++;
        if (this.frameCount % 60 === 0) {
            this.fpsElement.textContent = `FPS: ${Math.round(1 / deltaTime)}`;
            this.particlesElement.textContent = `Particles: ${this.numParticles}`;
        }

        requestAnimationFrame((time) => this.animate(time));
    }
}

window.addEventListener('load', () => {
    new ParticleSimulation();
});
