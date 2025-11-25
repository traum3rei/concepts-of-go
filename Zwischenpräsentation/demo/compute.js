async function runComputePipeline() {
    const outputDiv = document.getElementById('computeOutput');
    
    try {
        // Schritt 1: WebGPU Adapter und Device anfordern
        // Der Adapter repräsentiert die physische GPU
        if (!navigator.gpu) {
            throw new Error('WebGPU wird von diesem Browser nicht unterstützt!');
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('Kein WebGPU Adapter gefunden!');
        }
        
        // Das Device ist die Schnittstelle zur GPU
        const device = await adapter.requestDevice();
        
        // Schritt 2: Compute Shader definieren
        // WGSL (WebGPU Shading Language) ist die Shader-Sprache für WebGPU
        // Dieser Shader verdoppelt einfach jede Zahl im Buffer
        const computeShaderCode = `
            // @group(0) @binding(0) bedeutet: Gruppe 0, Binding-Position 0
            // var<storage, read_write> erlaubt Lese- und Schreibzugriff
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            
            // @compute @workgroup_size(64) definiert einen Compute Shader
            // mit 64 Threads pro Workgroup
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                // global_id.x ist die eindeutige ID dieses Threads
                let index = global_id.x;
                
                // Sicherheitscheck: Nur verarbeiten, wenn Index innerhalb des Arrays liegt
                if (index < arrayLength(&data)) {
                    // Verdopple den Wert an dieser Position
                    data[index] = data[index] * 2.0;
                }
            }
        `;
        
        // Schritt 3: Shader-Modul erstellen
        const computeModule = device.createShaderModule({
            label: 'Compute Shader',
            code: computeShaderCode
        });
        
        // Schritt 4: Eingabedaten vorbereiten
        // Wir erstellen ein Array mit Zahlen 1-10
        const inputData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        
        // Schritt 5: GPU Buffer erstellen
        // Der Buffer muss groß genug sein für unsere Daten
        const bufferSize = inputData.byteLength;
        const buffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE |      // Für Compute Shader
                   GPUBufferUsage.COPY_SRC |     // Zum Lesen zurück
                   GPUBufferUsage.COPY_DST       // Zum Schreiben
        });
        
        // Schritt 6: Daten auf die GPU kopieren
        device.queue.writeBuffer(buffer, 0, inputData);
        
        // Schritt 7: Compute Pipeline erstellen
        // Die Pipeline definiert, welcher Shader verwendet wird
        // WICHTIG: Wir erstellen die Pipeline zuerst, um das Bind Group Layout zu erhalten
        const computePipeline = device.createComputePipeline({
            label: 'Compute Pipeline',
            layout: 'auto',  // Automatisches Layout
            compute: {
                module: computeModule,
                entryPoint: 'main'
            }
        });
        
        // Schritt 8: Bind Group erstellen
        // Bind Groups verbinden Buffers mit Shader-Bindings
        // WICHTIG: Wenn wir 'auto' Layout verwenden, müssen wir das Layout von der Pipeline holen
        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),  // Hole Layout von der Pipeline
            entries: [{
                binding: 0,
                resource: {
                    buffer: buffer
                }
            }]
        });
        
        // Schritt 9: Command Encoder erstellen
        // Command Encoder zeichnen Befehle auf, die später ausgeführt werden
        const encoder = device.createCommandEncoder({
            label: 'Compute Command Encoder'
        });
        
        // Schritt 10: Compute Pass erstellen und ausführen
        const pass = encoder.beginComputePass();
        pass.setPipeline(computePipeline);
        pass.setBindGroup(0, bindGroup);
        
        // Dispatch: Starte genug Workgroups, um alle Daten zu verarbeiten
        // Wir haben 10 Elemente, Workgroup-Größe ist 64
        // Also brauchen wir 1 Workgroup (10 < 64)
        const workgroupCount = Math.ceil(inputData.length / 64);
        pass.dispatchWorkgroups(workgroupCount);
        pass.end();
        
        // Schritt 11: Command Buffer erstellen und ausführen
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
        
        // WICHTIG: Warte darauf, dass die GPU die Berechnung abgeschlossen hat
        // Die Befehle werden asynchron ausgeführt, daher müssen wir warten
        await device.queue.onSubmittedWorkDone();
        
        // Schritt 12: Ergebnis zurücklesen
        // Erstelle einen Buffer zum Lesen
        const readBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        // Kopiere Daten vom Compute Buffer zum Read Buffer
        const readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, bufferSize);
        device.queue.submit([readEncoder.finish()]);
        
        // Warte darauf, dass der Kopiervorgang abgeschlossen ist
        await device.queue.onSubmittedWorkDone();
        
        // Warte auf Fertigstellung und mappe den Buffer
        await readBuffer.mapAsync(GPUMapMode.READ);
        const mappedRange = readBuffer.getMappedRange();
        const result = new Float32Array(mappedRange);
        
        // WICHTIG: Kopiere die Daten in ein neues Array, BEVOR wir unmap() aufrufen
        // Nach unmap() ist der ArrayBuffer "detached" und nicht mehr zugänglich
        const resultCopy = new Float32Array(result);
        readBuffer.unmap();
        
        outputDiv.textContent += '\n\n📊 ERGEBNIS:';
        outputDiv.textContent += '\nEingabe: ' + Array.from(inputData).join(', ');
        outputDiv.textContent += '\nAusgabe: ' + Array.from(resultCopy).join(', ');
        
    } catch (error) {
        outputDiv.className = 'error';
        outputDiv.textContent = '❌ Fehler: ' + error.message;
        console.error(error);
    }
}

