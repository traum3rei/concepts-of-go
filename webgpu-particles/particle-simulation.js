class ParticleSimulation {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.particleBuffer = null;
        this.uniformBuffer = null;
        this.bindGroup = null;
        this.numParticles = 10000;
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
                    color: vec3<f32>,
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
                    
                    // Gravity
                    force.y -= uniforms.gravity;
                    
                    // Fluid pressure and viscosity forces
                    var pressure = 0.0;
                    var viscosity = vec2<f32>(0.0, 0.0);
                    
                    // Calculate forces from nearby particles
                    for (var i = 0u; i < arrayLength(&particles); i++) {
                        if (i == index) {
                            continue;
                        }
                        
                        let other = particles[i];
                        let dx = particle.position.x - other.position.x;
                        let dy = particle.position.y - other.position.y;
                        let distance = sqrt(dx * dx + dy * dy);
                        
                        if (distance < 20.0 && distance > 0.001) {
                            // Pressure force (repulsion)
                            let pressureForce = 100.0 / (distance * distance);
                            let pressureX = (dx / distance) * pressureForce;
                            let pressureY = (dy / distance) * pressureForce;
                            force.x += pressureX;
                            force.y += pressureY;
                            
                            // Viscosity force (attraction/damping)
                            let viscosityForce = 0.1;
                            let velDiffX = other.velocity.x - particle.velocity.x;
                            let velDiffY = other.velocity.y - particle.velocity.y;
                            force.x += velDiffX * viscosityForce;
                            force.y += velDiffY * viscosityForce;
                        }
                    }
                    
                    // Apply forces to velocity
                    particles[index].velocity += force * uniforms.deltaTime;
                    
                    // Update position
                    particles[index].position += particles[index].velocity * uniforms.deltaTime;
                    
                    // Boundary collision with damping
                    if (particles[index].position.x < 10.0 || particles[index].position.x > uniforms.width - 10.0) {
                        particles[index].velocity.x *= -0.3;
                        particles[index].position.x = clamp(particles[index].position.x, 10.0, uniforms.width - 10.0);
                    }
                    if (particles[index].position.y < 10.0 || particles[index].position.y > uniforms.height - 10.0) {
                        particles[index].velocity.y *= -0.3;
                        particles[index].position.y = clamp(particles[index].position.y, 10.0, uniforms.height - 10.0);
                    }
                    
                    // Damping for stability
                    particles[index].velocity *= 0.99;
                }
            `
        });

        this.vertexShader = this.device.createShaderModule({
            code: `
                struct Particle {
                    position: vec2<f32>,
                    velocity: vec2<f32>,
                    color: vec3<f32>,
                }

                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) color: vec3<f32>,
                }

                @group(0) @binding(0) var<storage, read> particles: array<Particle>;

                @vertex
                fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                    let particleIndex = vertexIndex / 6u; // 6 vertices per particle (2 triangles)
                    let vertexInParticle = vertexIndex % 6u;
                    
                    let particle = particles[particleIndex];
                    let size = 4.0; // Particle size
                    
                    // Create two triangles to form a quad for each particle
                    let offsets = array<vec2<f32>, 6>(
                        // Triangle 1: bottom-left, bottom-right, top-left
                        vec2<f32>(-size, -size), // 0: bottom-left
                        vec2<f32>( size, -size), // 1: bottom-right  
                        vec2<f32>(-size,  size), // 2: top-left
                        // Triangle 2: bottom-right, top-right, top-left
                        vec2<f32>( size, -size), // 3: bottom-right
                        vec2<f32>( size,  size), // 4: top-right
                        vec2<f32>(-size,  size)  // 5: top-left
                    );
                    
                    let offset = offsets[vertexInParticle];
                    let worldPos = particle.position + offset;
                    
                    // Convert to clip space (0,0 to 800,600 -> -1,1 to 1,-1)
                    let clipPos = vec2<f32>(
                        (worldPos.x / 400.0) - 1.0,
                        1.0 - (worldPos.y / 300.0)
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
                fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
                    return vec4<f32>(color, 1.0);
                }
            `
        });
    }

    async setupBuffers() {
        
        const particleSize = 7 * 4; 
        this.particleBuffer = this.device.createBuffer({
            size: this.numParticles * particleSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Initialize particles as a water blob for fluid simulation (This is kinda scuffed ngl)
        const particleData = new Float32Array(this.numParticles * 7);
        const centerX = 400;
        const centerY = 300;
        const blobRadius = 80;
        
        for (let i = 0; i < this.numParticles; i++) {
            const baseIndex = i * 7;
            
            // Create a circular blob of particles
            const angle = (i / this.numParticles) * 2 * Math.PI;
            const radius = Math.random() * blobRadius;
            particleData[baseIndex + 0] = centerX + Math.cos(angle) * radius; // x
            particleData[baseIndex + 1] = centerY + Math.sin(angle) * radius; // y
            particleData[baseIndex + 2] = 0; // vx 
            particleData[baseIndex + 3] = 0; // vy 
            
            // Water-like blue colors
            particleData[baseIndex + 4] = 0.1 + Math.random() * 0.3; // r 
            particleData[baseIndex + 5] = 0.3 + Math.random() * 0.4; // g
            particleData[baseIndex + 6] = 0.7 + Math.random() * 0.3; // b 
        }

        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);

        
        this.uniformBuffer = this.device.createBuffer({
            size: 32, // 8 floats * 4 bytes (with padding for alignment)
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
            ],
        });
    }

    updateUniforms(deltaTime) {
        const uniformData = new Float32Array([
            deltaTime,
            800.0, // width
            600.0, // height
            50.0,  // gravity 
            0.0,   // padding
            0.0,   // padding
            0.0,   // padding
            0.0,   // padding
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
