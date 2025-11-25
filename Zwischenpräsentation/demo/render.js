async function runRenderPipeline() {
    const canvas = document.getElementById('renderCanvas');
    
    try {
        // Schritt 1: WebGPU Context vom Canvas erhalten
        const context = canvas.getContext('webgpu');
        if (!context) {
            throw new Error('WebGPU Context konnte nicht erstellt werden!');
        }
        
        // Schritt 2: Adapter und Device anfordern
        if (!navigator.gpu) {
            throw new Error('WebGPU wird von diesem Browser nicht unterstützt!');
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('Kein WebGPU Adapter gefunden!');
        }
        
        const device = await adapter.requestDevice();
        
        // Schritt 3: Canvas Format bestimmen
        // Das Format muss mit dem Display kompatibel sein
        const format = navigator.gpu.getPreferredCanvasFormat();
        
        // Schritt 4: Context konfigurieren
        context.configure({
            device: device,
            format: format
        });
        
        // Schritt 5: Vertex Shader definieren
        // Vertex Shader verarbeitet jeden Vertex des Dreiecks
        const vertexShaderCode = `
            // Vertex Position als Input
            struct VertexInput {
                @location(0) position: vec2<f32>,
                @location(1) color: vec3<f32>,
            };
            
            // Output an Fragment Shader
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec3<f32>,
            };
            
            // @vertex markiert dies als Vertex Shader
            @vertex
            fn main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                // Position im Clip-Space (von -1 bis 1)
                output.position = vec4<f32>(input.position, 0.0, 1.0);
                // Farbe weitergeben
                output.color = input.color;
                return output;
            }
        `;

        
        // Schritt 6: Fragment Shader definieren
        // Fragment Shader bestimmt die Farbe jedes Pixels
        const fragmentShaderCode = `
            // Input vom Vertex Shader
            struct FragmentInput {
                @location(0) color: vec3<f32>,
            };
            
            // @fragment markiert dies als Fragment Shader
            @fragment
            fn main(input: FragmentInput) -> @location(0) vec4<f32> {
                // Gib die interpolierte Farbe zurück
                return vec4<f32>(input.color, 1.0);
            }
        `;
        
        // Schritt 7: Shader Module erstellen
        const vertexModule = device.createShaderModule({
            label: 'Vertex Shader',
            code: vertexShaderCode
        });
        
        const fragmentModule = device.createShaderModule({
            label: 'Fragment Shader',
            code: fragmentShaderCode
        });
        
        // Schritt 8: Vertex Daten definieren
        // Ein Dreieck mit 3 Vertices
        // Jeder Vertex hat: Position (x, y) und Farbe (r, g, b)
        const vertices = new Float32Array([
            // Vertex 1: Oben, rot
             0.0,  1, 1.0, 0.0, 0.0,
            // Vertex 2: Links unten, grün
            -1, -1, 0.0, 1.0, 0.0,
            // Vertex 3: Rechts unten, blau
             1, -1, 0.0, 0.0, 1.0,
        ])
        
        // Schritt 9: Vertex Buffer erstellen
        const vertexBuffer = device.createBuffer({
            label: 'Vertex Buffer',
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        
        // Daten auf die GPU kopieren
        device.queue.writeBuffer(vertexBuffer, 0, vertices);
        
        // Schritt 10: Render Pipeline erstellen
        // Die Pipeline definiert, wie gerendert wird
        const renderPipeline = device.createRenderPipeline({
            label: 'Render Pipeline',
            layout: 'auto',
            vertex: {
                module: vertexModule,
                entryPoint: 'main',
                buffers: [{
                    // Definiere das Layout der Vertex-Daten
                    arrayStride: 5 * 4,  // 5 Floats * 4 Bytes = 20 Bytes pro Vertex
                    attributes: [
                        {
                            // Position: Offset 0, 2 Floats
                            shaderLocation: 0,
                            offset: 0,
                            format: 'float32x2'
                        },
                        {
                            // Farbe: Offset 8 Bytes (2 Floats), 3 Floats
                            shaderLocation: 1,
                            offset: 2 * 4,
                            format: 'float32x3'
                        }
                    ]
                }]
            },
            fragment: {
                module: fragmentModule,
                entryPoint: 'main',
                targets: [{
                    format: format
                }]
            },
            primitive: {
                topology: 'triangle-list'  // Zeichne Dreiecke
            }
        });
        
        // Schritt 11: Render Loop
        function render() {
            // Command Encoder erstellen
            const encoder = device.createCommandEncoder({
                label: 'Render Command Encoder'
            });
            
            // Texture View vom Canvas erhalten
            const view = context.getCurrentTexture().createView();
            
            // Render Pass erstellen
            const renderPass = encoder.beginRenderPass({
                label: 'Render Pass',
                colorAttachments: [{
                    view: view,
                    clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },  // Dunkelgrauer Hintergrund
                    loadOp: 'clear',  // Lösche den Buffer
                    storeOp: 'store'  // Speichere das Ergebnis
                }]
            });
            
            // Pipeline setzen
            renderPass.setPipeline(renderPipeline);
            
            // Vertex Buffer binden
            renderPass.setVertexBuffer(0, vertexBuffer);
            
            // Zeichne 3 Vertices (ein Dreieck)
            renderPass.draw(3);
            
            // Render Pass beenden
            renderPass.end();
            
            // Command Buffer erstellen und ausführen
            const commandBuffer = encoder.finish();
            device.queue.submit([commandBuffer]);
        }
        
        // Einmal rendern
        render();
        
    } catch (error) {
        console.error(error);
    }
}

