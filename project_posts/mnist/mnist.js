"use strict";

// ONNX variables
let session = null;

// Drawing canvas variables
const canvas = document.getElementById('canvas');
const canvasCtx = canvas.getContext('2d');
let isDrawing = false;

showLoadingScreen();

// Histogram variables
const histogram = document.querySelector("#histogram");
const histogramCtx = histogram.getContext("2d");

async function init() {
    try {
        session = await ort.InferenceSession.create('./mnist.onnx');
        clearCanvas();  // clear when done loading
    }
    catch (e) {
        console.error(e);
    }
}

function prepareImageDataForInference() {
    const dims = [1, 1, 28, 28];
    let imgData = canvasCtx.getImageData(0, 0, canvas.width, canvas.height).data;
    let floatData = new Float32Array(28 * 28);
    for (let i = 0; i < imgData.length / 4; i++) {
        let colorValue = 1 - imgData[4 * i] / 255.0; // Subtracting s.t. color value is inversed
        floatData[i] = colorValue;
    }
    let tensor = { input1: new ort.Tensor('float32', floatData, dims) };
    return tensor;
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

async function runInference() {
    if (!session) {
        console.warn('Model not done loading!');
        return;
    }
    const feed = prepareImageDataForInference();
    try {
        const results = await session.run(feed);
        const resultsAfterSoftmax = softmax(results.output1.data);
        // console.log(feed);
        drawHistogram(resultsAfterSoftmax);
    }
    catch (e) {
        console.error('Error during inference:', e);
    }
}

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    canvasCtx.beginPath();
    runInference();     // running model
}

function draw(e) {
    if (!isDrawing) return;

    const canvasRect = canvas.getBoundingClientRect();

    const scaleX = canvas.width / canvasRect.width;
    const scaleY = canvas.height / canvasRect.height;

    const x = (e.clientX - canvasRect.left) * scaleX;
    const y = (e.clientY - canvasRect.top) * scaleY;

    canvasCtx.lineWidth = 2;
    canvasCtx.lineCap = 'round';
    canvasCtx.strokeStyle = 'black';

    canvasCtx.lineTo(x, y);
    canvasCtx.stroke();
    canvasCtx.beginPath();
    canvasCtx.moveTo(x, y);
}

function handleStart(e) {
    isDrawing = true;
    e.preventDefault();

    const canvasRect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / canvasRect.width;
    const scaleY = canvas.height / canvasRect.height;
    const x = (e.touches[0].clientX - canvasRect.left) * scaleX;
    const y = (e.touches[0].clientY - canvasRect.top) * scaleY;

    canvasCtx.beginPath();
    canvasCtx.moveTo(x, y);
}

function handleMove(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const canvasRect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / canvasRect.width;
    const scaleY = canvas.height / canvasRect.height;
    const x = (e.touches[0].clientX - canvasRect.left) * scaleX;
    const y = (e.touches[0].clientY - canvasRect.top) * scaleY;

    canvasCtx.lineWidth = 2;
    canvasCtx.lineCap = 'round';
    canvasCtx.strokeStyle = 'black';

    canvasCtx.lineTo(x, y);
    canvasCtx.stroke();
    canvasCtx.beginPath();
    canvasCtx.moveTo(x, y);
}

function clearCanvas() {
    canvasCtx.fillStyle = 'white';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    clearHistogram();
}

function clearHistogram() {
    // Variables
    const backgroundColor = '#145252';
    const textColor = '#f9fad5';
    const seperationLineColor = '#f9fad5';
    const textExtraYmargin = 3;
    const lineMarginToText = 25;
    // Draw blank background
    histogramCtx.fillStyle = backgroundColor;
    histogramCtx.fillRect(0, 0, histogram.width, histogram.height);
    // Draw y-axis classification numbers
    histogramCtx.fillStyle = textColor;
    histogramCtx.font = 'italic 15pt Calibri';
    histogramCtx.textBaseline = 'top';
    const textXPos = 0;
    for (let i = 0; i < 10; i++) {
        const textYPos = histogram.height * i / 10 + textExtraYmargin;
        histogramCtx.fillText(i, textXPos, textYPos);
    }
    // Draw vertical seperation line
    histogramCtx.strokeStyle = seperationLineColor;
    histogramCtx.beginPath();
    histogramCtx.moveTo(lineMarginToText, 0);
    histogramCtx.lineTo(lineMarginToText, histogram.height);
    histogramCtx.stroke();
}

function drawHistogram(data) {
    clearHistogram();
    histogramCtx.save();     // why save?
    histogramCtx.font = 'italic 10pt Calibri';
    // histogramCtx.fillStyle = barcolor;

    const textColor = '#f9fad5';
    const barColor = '#f9fad5';
    const barThicknessReduction = 3;
    const barWidth = histogram.height/10 - 2 * barThicknessReduction;
    const canvasSize = histogram.height;     // same as width
    const barMarginToText = 40;
    const likelihoodMarginToBar = 5;
    const likelihoodYShift = 6;

    const backgroundColor = '#145252';
    const rectangleLineWidth = 3;

    var posX = barMarginToText;
    var mostLikelyNumber = data.indexOf(Math.max(...data));
    for (var i=0; i < data.length; i++) {

        let barLength = data[i] * canvasSize;
        let posY = i * canvasSize/10;
        histogramCtx.fillStyle = barColor;
        // histogramCtx.lineWidth = 1;
        if (i === mostLikelyNumber) {
            // Draw bar for most likely classification
            histogramCtx.fillRect(
                posX, 
                posY + barThicknessReduction, 
                barLength, 
                barWidth
            );
        }
        else {
            // Draw bar for other classifications
            histogramCtx.fillRect(
                posX, 
                posY + barThicknessReduction, 
                barLength, 
                barWidth
            );
            histogramCtx.fillStyle = backgroundColor;
            histogramCtx.fillRect(
                posX + rectangleLineWidth, 
                posY + barThicknessReduction + rectangleLineWidth, 
                barLength - 2 * rectangleLineWidth, 
                barWidth - 2 * rectangleLineWidth
            );
        }
        histogramCtx.save();

        // Draw text
        const classLikelihood = data[i].toFixed(2);
        histogramCtx.fillStyle = textColor;
        histogramCtx.fillText(
            classLikelihood, 
            posX + barLength + likelihoodMarginToBar, 
            posY + likelihoodYShift
        );
        histogramCtx.restore();
    }
    histogramCtx.restore();
}

function showLoadingScreen() {
    // TODO: make better looking loading screen
    // Use a different canvas with higher resolution for the loading screen,
    // swapping it out when the model is loaded.
    canvasCtx.fillStyle = '#AAAAAA';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.font = "8px sans-serif";
    canvasCtx.fillStyle = '#000000';
    canvasCtx.textAlign = 'center';
    canvasCtx.fillText("Loading", 14, 18);
}

// Add event listeners for mouse
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', () => {if (isDrawing) stopDrawing(); });

// Add event listeners for touch
canvas.addEventListener('touchstart', handleStart, { passive: false });
canvas.addEventListener('touchmove', handleMove, { passive: false });
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchcancel', stopDrawing);

init();