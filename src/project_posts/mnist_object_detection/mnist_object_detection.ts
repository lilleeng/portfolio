interface OrtTensor {
    data: Float32Array | Int32Array | Uint8Array;
    dims: number[];
}

interface OrtSession {
    inputNames: string[];
    outputNames: string[];
    run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

interface OrtRuntime {
    InferenceSession: { create(path: string): Promise<OrtSession> };
    Tensor: new (type: string, data: Float32Array, dims: number[]) => OrtTensor;
}

declare const ort: OrtRuntime;

interface Detection {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    score: number;
    label: number;
}

function requiredElement<T extends Element>(selector: string): T {
    const element = document.querySelector<T>(selector);
    if (!element) {
        throw new Error(`Object detection page is missing ${selector}.`);
    }
    return element;
}

const drawingCanvas = requiredElement<HTMLCanvasElement>("#drawing-canvas");
const overlayCanvas = requiredElement<HTMLCanvasElement>("#overlay-canvas");
const clearButton = requiredElement<HTMLButtonElement>("#clear-button");
// const statusElement = requiredElement<HTMLParagraphElement>("#status");
const resultsText = requiredElement<HTMLParagraphElement>("#results-text");
const detectionsList = requiredElement<HTMLUListElement>("#detections");
const drawingContextElement = drawingCanvas.getContext("2d");
const overlayContextElement = overlayCanvas.getContext("2d");

if (!drawingContextElement || !overlayContextElement) {
    throw new Error("Could not create the canvas contexts.");
}

const drawingContext = drawingContextElement;
const overlayContext = overlayContextElement;
const canvasSize = 64;
const displaySize = 480;
const gridSize = 8;
const valuesPerCell = 15;   // Output logits?
const scoreThreshold = 0.35;
let session: OrtSession | null = null;
let isDrawing = false;

// function setStatus(message: string): void {
//     statusElement.textContent = message;
// }

function clearOverlay(): void {
    overlayContext.clearRect(0, 0, displaySize, displaySize);
}

function clearCanvas(): void {
    drawingContext.fillStyle = "white";
    drawingContext.fillRect(0, 0, canvasSize, canvasSize);
    clearOverlay();
    resultsText.textContent = "No detections yet.";
    detectionsList.replaceChildren();
    // setStatus(session ? "Model ready. Draw a digit." : "Loading model...");
}

function canvasPoint(event: PointerEvent): { x: number; y: number } {
    const rectangle = drawingCanvas.getBoundingClientRect();
    return {
        x: (event.clientX - rectangle.left) * canvasSize / rectangle.width,
        y: (event.clientY - rectangle.top) * canvasSize / rectangle.height
    };
}

function startDrawing(event: PointerEvent): void {
    isDrawing = true;
    drawingCanvas.setPointerCapture(event.pointerId);
    const point = canvasPoint(event);
    drawingContext.beginPath();
    drawingContext.moveTo(point.x, point.y);
    drawingContext.lineWidth = 2;
    drawingContext.lineCap = "round";
    drawingContext.strokeStyle = "black";
}

function draw(event: PointerEvent): void {
    if (!isDrawing) {
        return;
    }
    const point = canvasPoint(event);
    drawingContext.lineTo(point.x, point.y);
    drawingContext.stroke();
    drawingContext.beginPath();
    drawingContext.moveTo(point.x, point.y);
}

function stopDrawing(): void {
    if (!isDrawing) {
        return;
    }
    isDrawing = false;
    drawingContext.beginPath();
    void runInference();
}

function prepareInput(): OrtTensor {
    if (!session) {
        throw new Error("The model is not loaded.");
    }
    const dimensions = [1, 1, canvasSize, canvasSize];
    const pixels = drawingContext.getImageData(0, 0, canvasSize, canvasSize).data;
    const values = new Float32Array(canvasSize * canvasSize);

    for (let index = 0; index < values.length; index++) {
        values[index] = 1 - pixels[index * 4] / 255;
    }
    return new ort.Tensor("float32", values, dimensions);
}

function decodeDetections(tensor: OrtTensor): Detection[] {
    const values = Array.from(tensor.data); // practically flattens the output tensor
    const detections: Detection[] = [];
    const sigmoid = (value: number): number => 1 / (1 + Math.exp(-value));

    for (let row = 0; row < gridSize; row++) {
        for (let column = 0; column < gridSize; column++) {
            const offset = (row * gridSize + column) * valuesPerCell;

            const objectnessLogit = values[offset];
            const objectness = sigmoid(objectnessLogit);

            const centerX = values[offset + 1];     // in cell
            const centerY = values[offset + 2];     // in cell

            const width = (values[offset + 3] > 0) ? values[offset + 3] : 0;
            const height = (values[offset + 4] > 0) ? values[offset + 4] : 0;

            const classLogits = values.slice(offset + 5, offset + valuesPerCell);
            const maximumLogit = Math.max(...classLogits);
            const classExponentials = classLogits.map((logit) => Math.exp(logit - maximumLogit));   
                    // subtract maximumLogit for numerical stability, no change in result
            const exponentialSum = classExponentials.reduce((sum, value) => sum + value, 0);
            const classProbabilities = classExponentials.map((value) => value / exponentialSum);
            const label = classProbabilities.indexOf(Math.max(...classProbabilities));
            const score = objectness * classProbabilities[label];

            const x1 = (column + centerX - width / 2) / gridSize;
            const y1 = (row + centerY - height / 2) / gridSize;
            const x2 = (column + centerX + width / 2) / gridSize;
            const y2 = (row + centerY + height / 2) / gridSize;

            if (score >= scoreThreshold && x2 > x1 && y2 > y1) {
                detections.push({
                    x1: Math.max(0, x1),
                    y1: Math.max(0, y1),
                    x2: Math.min(1, x2),
                    y2: Math.min(1, y2),
                    score,
                    label
                });
            }
        }
    }
    return nonMaximumSuppression(detections);
}

function intersectionOverUnion(first: Detection, second: Detection): number {
    const intersectionWidth = Math.max(0, Math.min(first.x2, second.x2) - Math.max(first.x1, second.x1));
    const intersectionHeight = Math.max(0, Math.min(first.y2, second.y2) - Math.max(first.y1, second.y1));
    const intersection = intersectionWidth * intersectionHeight;
    const firstArea = (first.x2 - first.x1) * (first.y2 - first.y1);
    const secondArea = (second.x2 - second.x1) * (second.y2 - second.y1);
    return intersection / (firstArea + secondArea - intersection);
}

function nonMaximumSuppression(detections: Detection[]): Detection[] {
    const remaining = [...detections].sort((first, second) => second.score - first.score);
    const selected: Detection[] = [];
    while (remaining.length > 0) {
        const best = remaining.shift();
        if (!best) {
            break;
        }
        selected.push(best);
        for (let index = remaining.length - 1; index >= 0; index--) {
            if (remaining[index].label === best.label &&
                intersectionOverUnion(best, remaining[index]) > 0.5) {
                remaining.splice(index, 1);
            }
        }
    }
    return selected;
}

function drawDetections(detections: Detection[]): void {
    clearOverlay();
    overlayContext.strokeStyle = "#d22";
    overlayContext.lineWidth = 2;
    overlayContext.font = "20px sans-serif";

    for (const detection of detections) {
        const x1 = Math.round(detection.x1 * canvasSize) * (displaySize / canvasSize);
        const y1 = Math.round(detection.y1 * canvasSize) * (displaySize / canvasSize);
        const x2 = Math.round(detection.x2 * canvasSize) * (displaySize / canvasSize);
        const y2 = Math.round(detection.y2 * canvasSize) * (displaySize / canvasSize);
        const label = `${detection.label} (${(detection.score * 100).toFixed(1)}%)`;
        const labelWidth = overlayContext.measureText(label).width + 12;
        const labelHeight = 28;
        const labelY = Math.max(labelHeight, y1);

        overlayContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
        overlayContext.fillStyle = "rgba(210, 34, 34, 0.7)";
        overlayContext.fillRect(x1, labelY - labelHeight, labelWidth, labelHeight);
        overlayContext.fillStyle = "white";
        overlayContext.fillText(label, x1 + 6, labelY - 7);
    }
}

async function runInference(): Promise<void> {
    if (!session) {
        // setStatus("The model is still loading.");
        return;
    }
    try {
        // setStatus("Detecting...");
        const output = (await session.run({
            [session.inputNames[0]]: prepareInput()
        }))[session.outputNames[0]];
        const detections = decodeDetections(output);
        drawDetections(detections);
        resultsText.textContent = detections.length
            ? `${detections.length} object(s) found.`
            : "No objects found above the confidence threshold.";
        detectionsList.replaceChildren(...detections.map((detection) => {
            const item = document.createElement("li");
            item.textContent = `Class ${detection.label}: ${(detection.score * 100).toFixed(1)}%`;
            return item;
        }));
        // setStatus("Detection complete.");
    } catch (error) {
        console.error("Object detection failed:", error);
        // setStatus("Detection failed. Check the browser console for details.");
    }
}

drawingCanvas.addEventListener("pointerdown", startDrawing);
drawingCanvas.addEventListener("pointermove", draw);
drawingCanvas.addEventListener("pointerup", stopDrawing);
drawingCanvas.addEventListener("pointercancel", stopDrawing);
clearButton.addEventListener("click", clearCanvas);

clearCanvas();
void (async () => {
    try {
        session = await ort.InferenceSession.create("./mnist_object_detection.onnx");
        // setStatus("Model ready. Draw a digit.");
    } catch (error) {
        console.error("Could not load object detection model:", error);
        // setStatus("Could not load the model. Check the browser console for details.");
    }
})();
