// DOM Elements
const videoElement = document.getElementById('input-video');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const sentenceElement = document.getElementById('sentence');
const predictionsContainer = document.getElementById('predictions-container');
const statusElement = document.getElementById('status');
const clearBtn = document.getElementById('clear-btn');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const toggleCameraBtn = document.getElementById('toggle-camera-btn');
const toggleLandmarksBtn = document.getElementById('toggle-landmarks-btn');

// Initialize Socket.io connection
const socket = io();

// Variables
let camera = null;
let holistic = null;
let isRunning = false;
let keypoints = [];
let lastSentKeypoints = 0;
let sentence = [];
let predictions = [];
let isSocketConnected = false;
let currentFacingMode = 'user'; // Default to front camera
let isFullscreen = false;
let showLandmarks = true; // Default to showing landmarks
const threshold = 0.5;
const sequenceLength = 120; // Number of frames to collect before prediction

// Initialize canvas when page loads
function initializeCanvas() {
    // Make canvas responsive to screen width
    resizeCanvasToDeviceWidth();
    
    // Listen for resize events
    window.addEventListener('resize', resizeCanvasToDeviceWidth);
}

// Update this function to use full viewport height
function resizeCanvasToDeviceWidth() {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    const videoContainer = document.querySelector('.video-container');
    
    if (videoContainer) {
        // Set container to full viewport dimensions
        videoContainer.style.width = '100%';
        videoContainer.style.height = '100vh';
        
        // Set canvas dimensions to match viewport
        canvasElement.width = screenWidth;
        canvasElement.height = screenHeight;
    }
}


// Generate colors for visualization
function generateColors(numColors) {
    const colors = [];
    for (let i = 0; i < numColors; i++) {
        const hue = Math.floor(360 * i / numColors);
        colors.push(`hsl(${hue}, 80%, 50%)`);
    }
    return colors;
}

// Toggle fullscreen mode
function toggleFullscreen() {
    const body = document.body;
    
    if (!isFullscreen) {
        // Enter fullscreen
        body.classList.add('fullscreen');
        if (body.requestFullscreen) {
            body.requestFullscreen();
        } else if (body.webkitRequestFullscreen) { /* Safari */
            body.webkitRequestFullscreen();
        } else if (body.msRequestFullscreen) { /* IE11 */
            body.msRequestFullscreen();
        }
        isFullscreen = true;
        fullscreenBtn.textContent = 'Exit Fullscreen';
        
        // Adjust canvas size for fullscreen
        resizeCanvasToFullscreen();
    } else {
        // Exit fullscreen
        body.classList.remove('fullscreen');
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) { /* Safari */
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) { /* IE11 */
            document.msExitFullscreen();
        }
        isFullscreen = false;
        fullscreenBtn.textContent = 'Fullscreen';
        
        // Reset canvas size
        const container = document.querySelector('.video-container');
        if (container) {
            canvasElement.width = container.clientWidth;
            canvasElement.height = container.clientHeight;
        }
    }
}

// Toggle landmarks visibility
function toggleLandmarksVisibility() {
    showLandmarks = !showLandmarks;
    
    if (showLandmarks) {
        toggleLandmarksBtn.innerHTML = '<img src=".\\static\\icons\\eye-on.svg" alt="Hide Landmarks" />';
        toggleLandmarksBtn.classList.remove('landmarks-hidden');
    } else {
        toggleLandmarksBtn.innerHTML = '<img src=".\\static\\icons\\eye-off.svg" alt="Show Landmarks" />';
        toggleLandmarksBtn.classList.add('landmarks-hidden');
    }
    
    // Update status text
    statusElement.textContent = `Status: Landmarks ${showLandmarks ? 'visible' : 'hidden'} - Running - Collected ${keypoints.length}/${sequenceLength} frames`;
}

// Update fullscreen function
function resizeCanvasToFullscreen() {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    
    canvasElement.width = screenWidth;
    canvasElement.height = screenHeight;
    
    // Ensure container takes full dimensions
    const container = document.querySelector('.video-container');
    if (container) {
        container.style.width = '100%';
        container.style.height = '100vh';
    }
    
    // If holistic is active, make sure we redraw
    if (holistic && isRunning) {
        holistic.send({image: videoElement});
    }
}

// Handle window resize
function onWindowResize() {
    if (isFullscreen) {
        resizeCanvasToFullscreen();
    } else {
        const container = document.querySelector('.video-container');
        if (container) {
            canvasElement.width = container.clientWidth;
            canvasElement.height = container.clientHeight;
        }
    }
}

// Initialize MediaPipe Holistic with callback
function initializeHolistic(onInitialized) {
    statusElement.textContent = 'Status: Initializing MediaPipe...';
    
    try {
        // Create holistic with proper path
        holistic = new Holistic({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
            }
        });
    
        // Configure holistic for better performance on mobile
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        holistic.setOptions({
            modelComplexity: isMobile ? 0 : 1, // Lower complexity for mobile
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
    
        holistic.onResults(onResults);
    
        // Better error handling during initialization
        setTimeout(() => {
            // Call the callback even if initialize() is problematic
            if (typeof onInitialized === 'function') {
                console.log("MediaPipe Holistic initialized (timeout fallback)");
                statusElement.textContent = 'Status: MediaPipe initialized';
                onInitialized();
            }
        }, 3000); // 3 second fallback
        
        // Try normal initialization
        console.log("Starting MediaPipe initialization");
        holistic.initialize()
            .then(() => {
                console.log("MediaPipe Holistic successfully initialized");
                statusElement.textContent = 'Status: MediaPipe Holistic initialized';
                
                // Call the callback when initialization is complete
                if (typeof onInitialized === 'function') {
                    onInitialized();
                }
            })
            .catch(error => {
                console.error("Error initializing MediaPipe:", error);
                statusElement.textContent = `Status: MediaPipe initialization warning - will try to continue`;
                
                // Still try to proceed if there's an error
                if (typeof onInitialized === 'function') {
                    setTimeout(() => onInitialized(), 1000);
                }
            });
    } catch (error) {
        console.error("Error creating MediaPipe instance:", error);
        statusElement.textContent = `Status: MediaPipe error - ${error.message}`;
        
        // Try to continue anyway after a delay
        if (typeof onInitialized === 'function') {
            setTimeout(() => onInitialized(), 2000);
        }
    }
}

// Extract keypoints from MediaPipe results
function extractKeypoints(results) {
    // Initialize arrays for each part
    const pose = results.poseLandmarks ? Array.from(results.poseLandmarks).map(lm => [lm.x, lm.y, lm.z, lm.visibility]) : Array(33).fill([0, 0, 0, 0]);
    const face = Array(468).fill([0, 0, 0]); // Face is optional in this app
    const leftHand = results.leftHandLandmarks ? Array.from(results.leftHandLandmarks).map(lm => [lm.x, lm.y, lm.z]) : Array(21).fill([0, 0, 0]);
    const rightHand = results.rightHandLandmarks ? Array.from(results.rightHandLandmarks).map(lm => [lm.x, lm.y, lm.z]) : Array(21).fill([0, 0, 0]);
    
    // Flatten arrays
    const flattenedPose = pose.flat();
    const flattenedLeftHand = leftHand.flat();
    const flattenedRightHand = rightHand.flat();
    
    // Combine all keypoints into a single flattened array
    return [...flattenedPose, ...flattenedLeftHand, ...flattenedRightHand];
}

// Update this function to fill the entire screen
function adjustCanvasToScreenWidth() {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    
    // Get actual video dimensions if available
    let videoWidth = videoElement.videoWidth;
    let videoHeight = videoElement.videoHeight;
    
    if (videoWidth && videoHeight) {
        // Always use full viewport dimensions
        canvasElement.width = screenWidth;
        canvasElement.height = screenHeight;
        
        // Update container to full viewport
        const container = document.querySelector('.video-container');
        if (container) {
            container.style.width = '100%';
            container.style.height = '100vh';
        }
    } else {
        // If no video dimensions yet, use viewport dimensions
        canvasElement.width = screenWidth;
        canvasElement.height = screenHeight;
        
        const container = document.querySelector('.video-container');
        if (container) {
            container.style.width = '100%';
            container.style.height = '100vh';
        }
    }
}

// Modified drawImage function to respect aspect ratio
function onResults(results) {
    if (!isRunning) return;
    
    // Ensure video container and canvas match screen width
    adjustCanvasToScreenWidth();
    
    // Clear the canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Calculate proper dimensions to maintain aspect ratio
    const imgWidth = results.image.width;
    const imgHeight = results.image.height;
    const canvasWidth = canvasElement.width;
    const canvasHeight = canvasElement.height;
    
    // Calculate dimensions while maintaining aspect ratio
    let drawWidth, drawHeight, offsetX, offsetY;
    
    const imgAspect = imgWidth / imgHeight;
    const canvasAspect = canvasWidth / canvasHeight;
    
    if (imgAspect > canvasAspect) {
        // Image is wider than canvas (relative to their heights)
        drawWidth = canvasWidth;
        drawHeight = canvasWidth / imgAspect;
        offsetX = 0;
        offsetY = (canvasHeight - drawHeight) / 2;
    } else {
        // Image is taller than canvas (relative to their widths)
        drawHeight = canvasHeight;
        drawWidth = canvasHeight * imgAspect;
        offsetX = (canvasWidth - drawWidth) / 2;
        offsetY = 0;
    }
    
    // Draw the camera feed on the canvas while maintaining aspect ratio
    canvasCtx.drawImage(
        results.image, 
        offsetX, offsetY, 
        drawWidth, drawHeight
    );
    
    // Only draw landmarks if showLandmarks is true
    if (showLandmarks) {
        drawLandmarks(canvasCtx, results);
    }
    
    // Extract keypoints and add to sequence
    const keypointsData = extractKeypoints(results);
    keypoints.push(keypointsData);
    
    // Keep only the last 120 frames
    if (keypoints.length > sequenceLength) {
        keypoints.shift();
    }
    
    // When we have 120 frames and enough time has passed since last prediction
    if (keypoints.length === sequenceLength && 
        (Date.now() - lastSentKeypoints > 500)) { // Limit predictions to every 500ms
        
        // Send keypoints to backend for prediction
        predictSign();
        lastSentKeypoints = Date.now();
    }
    
    // Update status if still in running state
    if (isRunning) {
        statusElement.textContent = `Status: Running - Collected ${keypoints.length}/${sequenceLength} frames ${showLandmarks ? '' : '(landmarks hidden)'}`;
    }
    
    canvasCtx.restore();
}

// Update fullscreen function
function resizeCanvasToFullscreen() {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    
    canvasElement.width = screenWidth;
    canvasElement.height = screenHeight;
    
    // Ensure container takes full dimensions
    const container = document.querySelector('.video-container');
    if (container) {
        container.style.width = '100%';
        container.style.height = '100vh';
    }
    
    // If holistic is active, make sure we redraw
    if (holistic && isRunning) {
        holistic.send({image: videoElement});
    }
}

// Draw landmarks on canvas
function drawLandmarks(ctx, results) {
    try {
        // Make sure dependencies are available
        if (!window.drawConnectors || !window.drawLandmarks) {
            console.warn("MediaPipe drawing utilities not available");
            return;
        }
        
        // Draw pose connections
        if (results.poseLandmarks) {
            window.drawConnectors(ctx, results.poseLandmarks, window.POSE_CONNECTIONS,
                          {color: '#00FF00', lineWidth: 2});
            window.drawLandmarks(ctx, results.poseLandmarks,
                         {color: '#FF0000', lineWidth: 1});
        }
        
        // Draw left hand connections
        if (results.leftHandLandmarks) {
            window.drawConnectors(ctx, results.leftHandLandmarks, window.HAND_CONNECTIONS,
                          {color: '#CC0000', lineWidth: 2});
            window.drawLandmarks(ctx, results.leftHandLandmarks,
                         {color: '#00FF00', lineWidth: 1});
        }
        
        // Draw right hand connections
        if (results.rightHandLandmarks) {
            window.drawConnectors(ctx, results.rightHandLandmarks, window.HAND_CONNECTIONS,
                          {color: '#00CC00', lineWidth: 2});
            window.drawLandmarks(ctx, results.rightHandLandmarks,
                         {color: '#FF0000', lineWidth: 1});
        }
    } catch (error) {
        console.error("Error drawing landmarks:", error);
    }
}

// Send keypoints to backend for prediction using WebSockets
function predictSign() {
    try {
        if (!isSocketConnected) {
            statusElement.textContent = 'Status: Socket not connected, retrying...';
            // Fall back to HTTP if socket is not connected
            predictSignHttp();
            return;
        }
        
        statusElement.textContent = 'Status: Sending data via WebSocket...';
        
        // Send keypoints via WebSocket
        socket.emit('predict_sign', { keypoints: keypoints });
        
    } catch (error) {
        console.error('Error sending via WebSocket:', error);
        statusElement.textContent = `Status: WebSocket error - ${error.message}`;
        
        // Fall back to HTTP
        predictSignHttp();
    }
}

// Fallback HTTP method for predictions
async function predictSignHttp() {
    try {
        statusElement.textContent = 'Status: Falling back to HTTP...';
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ keypoints: keypoints })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        statusElement.textContent = 'Status: Prediction received via HTTP';
        
        // Process prediction result
        processPrediction(data);
    } catch (error) {
        console.error('Error predicting sign via HTTP:', error);
        statusElement.textContent = `Status: HTTP Error - ${error.message}`;
    }
}

// Process prediction from backend
function processPrediction(data) {
    if (data.error) {
        statusElement.textContent = `Status: Error - ${data.error}`;
        return;
    }
    
    // Update predictions array
    predictions.push(data.prediction);
    
    // Add to sentence if consistent predictions and confidence is high
    if (predictions.length >= 10) {
        const lastTenPredictions = predictions.slice(-10);
        const mostCommon = findMostCommon(lastTenPredictions);
        
        if (data.confidence > threshold && mostCommon === data.prediction) {
            // Only add if it's different from the last word
            if (sentence.length === 0 || sentence[sentence.length - 1] !== data.prediction) {
                sentence.push(data.prediction);
                updateSentence();
            }
        }
        
        // Keep the predictions array at a reasonable size
        if (predictions.length > 30) {
            predictions = predictions.slice(-30);
        }
    }
    
    // Visualize top predictions
    displayPredictions(data.top_predictions);
}

// Find the most common element in an array
function findMostCommon(arr) {
    const counts = {};
    let maxCount = 0;
    let maxItem = null;
    
    for (const item of arr) {
        counts[item] = (counts[item] || 0) + 1;
        if (counts[item] > maxCount) {
            maxCount = counts[item];
            maxItem = item;
        }
    }
    
    return maxItem;
}

// Update the sentence display
function updateSentence() {
    sentenceElement.textContent = sentence.join(' ');
}

// Display predictions as bars
function displayPredictions(topPredictions) {
    predictionsContainer.innerHTML = '';
    
    for (const pred of topPredictions) {
        const predDiv = document.createElement('div');
        predDiv.className = 'prediction-bar';
        
        const barDiv = document.createElement('div');
        barDiv.className = 'bar';
        barDiv.style.width = `${pred.probability * 200}px`;
        barDiv.textContent = pred.action;
        
        const probText = document.createElement('span');
        probText.textContent = pred.probability.toFixed(2);
        
        predDiv.appendChild(barDiv);
        predDiv.appendChild(probText);
        predictionsContainer.appendChild(predDiv);
    }
}

// Add this function to check if device has multiple cameras
async function hasMultipleCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        return videoDevices.length > 1;
    } catch (error) {
        console.error('Error checking for multiple cameras:', error);
        return false;
    }
}

// Toggle camera function with improved error handling and fallbacks
async function toggleCamera() {
    try {
        // Disable button during camera switch
        toggleCameraBtn.disabled = true;
        statusElement.textContent = `Status: Switching camera...`;
        
        // Toggle the facing mode
        currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
        
        // Stop current camera completely
        await stopCamera();
        
        // Try to start the new camera
        await startCamera();
        
        // Re-enable the button when done
        toggleCameraBtn.disabled = false;
        
    } catch (err) {
        console.error("Error switching camera:", err);
        statusElement.textContent = `Status: Camera error - ${err.message}`;
        
        // If switching failed, try without 'exact' constraint
        try {
            console.log("Trying alternative camera access method...");
            await startCameraWithFallback();
            toggleCameraBtn.disabled = false;
        } catch (fallbackErr) {
            console.error("Fallback camera access failed:", fallbackErr);
            // If all attempts fail, revert to the previous camera
            currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
            await startCamera();
            toggleCameraBtn.disabled = false;
        }
    }
}


// Improved startCamera function with better constraint handling
async function startCamera() {
    try {
        // Check if holistic is initialized
        if (!holistic) {
            console.error("MediaPipe Holistic not initialized");
            statusElement.textContent = "Status: Error - MediaPipe not initialized";
            return;
        }
        
        statusElement.textContent = `Status: Starting ${currentFacingMode === 'user' ? 'front' : 'back'} camera...`;
        
        // Get device screen dimensions
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        
        // Try with exact constraint first (more reliable on most devices)
        const constraints = {
            video: {
                facingMode: currentFacingMode === 'environment' ? 
                    { exact: "environment" } : "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };
        
        console.log("Using camera constraints:", JSON.stringify(constraints));
        
        // Request camera permission
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log(`${currentFacingMode} camera access granted`);
        
        // Log camera track details for debugging
        const videoTrack = stream.getVideoTracks()[0];
        console.log("Camera track:", videoTrack.label);
        console.log("Track settings:", videoTrack.getSettings());
        
        // Set canvas dimensions
        canvasElement.width = screenWidth;
        canvasElement.height = screenHeight;
        
        // Set container to full viewport
        const container = document.querySelector('.video-container');
        if (container) {
            container.style.width = '100%';
            container.style.height = '100vh';
        }
        
        // Stop this stream since Camera will create its own
        stream.getTracks().forEach(track => track.stop());
        
        // Create and start MediaPipe camera instance
        camera = new Camera(videoElement, {
            onFrame: async () => {
                if (holistic && isRunning) {
                    try {
                        await holistic.send({image: videoElement});
                    } catch (e) {
                        console.error("Error sending frame to MediaPipe:", e);
                    }
                }
            },
            facingMode: currentFacingMode,
            width: 1280,
            height: 720
        });
        
        console.log(`Starting ${currentFacingMode} camera...`);
        await camera.start();
        console.log("Camera started successfully");
        
        isRunning = true;
        statusElement.textContent = `Status: ${currentFacingMode === 'user' ? 'Front' : 'Back'} camera running`;
        
    } catch (error) {
        console.error(`Error starting ${currentFacingMode} camera:`, error);
        throw error; // Let the caller handle the error for fallback
    }
}

// Fallback method for camera access without 'exact' constraint
async function startCameraWithFallback() {
    try {
        // Alternative constraints without 'exact'
        const fallbackConstraints = {
            video: {
                facingMode: currentFacingMode,  // Simple string without exact
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };
        
        console.log("Using fallback constraints:", JSON.stringify(fallbackConstraints));
        
        // Try direct getUserMedia first
        const stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        const videoTrack = stream.getVideoTracks()[0];
        console.log("Fallback camera accessed:", videoTrack.label);
        
        // Stop this stream since Camera will create its own
        stream.getTracks().forEach(track => track.stop());
        
        // Create and start MediaPipe camera with simple facingMode
        camera = new Camera(videoElement, {
            onFrame: async () => {
                if (holistic && isRunning) {
                    try {
                        await holistic.send({image: videoElement});
                    } catch (e) {
                        console.error("Error sending frame to MediaPipe:", e);
                    }
                }
            },
            facingMode: currentFacingMode,
            width: 1280,
            height: 720
        });
        
        console.log("Starting fallback camera...");
        await camera.start();
        
        isRunning = true;
        statusElement.textContent = `Status: Camera running (fallback method)`;
        
    } catch (error) {
        console.error("Fallback camera access failed:", error);
        throw error;
    }
}

// Improved stopCamera function with better cleanup
async function stopCamera() {
    if (!camera) return Promise.resolve();
    
    isRunning = false;
    statusElement.textContent = 'Status: Stopping camera...';
    
    try {
        const oldCamera = camera;
        camera = null;
        
        await oldCamera.stop();
        console.log("Camera stopped successfully");
        
        // Additional cleanup for any remaining tracks
        const tracks = videoElement.srcObject?.getTracks() || [];
        tracks.forEach(track => {
            track.stop();
            console.log("Stopped additional track:", track.kind);
        });
        
        videoElement.srcObject = null;
        statusElement.textContent = 'Status: Camera stopped';
        
        // Short delay to ensure cleanup is complete
        await new Promise(resolve => setTimeout(resolve, 300));
        
        return Promise.resolve();
    } catch (error) {
        console.error('Error stopping camera:', error);
        statusElement.textContent = `Status: Error stopping camera - ${error.message}`;
        return Promise.reject(error);
    }
}


// Handle fullscreen change events
function handleFullscreenChange() {
    if (!document.fullscreenElement && 
        !document.webkitFullscreenElement && 
        !document.mozFullScreenElement && 
        !document.msFullscreenElement) {
        // Exited fullscreen via browser controls
        document.body.classList.remove('fullscreen');
        isFullscreen = false;
        fullscreenBtn.textContent = 'Fullscreen';
        const container = document.querySelector('.video-container');
        if (container) {
            canvasElement.width = container.clientWidth;
            canvasElement.height = container.clientHeight;
        }
    }
}

// Better device detection for camera availability
async function detectAvailableCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        console.log("Available video devices:", videoDevices.length);
        videoDevices.forEach((device, i) => {
            console.log(`Camera ${i+1}: ${device.label || 'unnamed device'}`);
        });
        
        // On some mobile devices, even with multiple cameras,
        // they might not be properly labeled until permission is granted
        if (videoDevices.length <= 1 || !videoDevices[0].label) {
            console.log("Limited camera info, trying to get permission first...");
            
            try {
                // Request basic video permission to get better device info
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                stream.getTracks().forEach(track => track.stop());
                
                // Try enumerating again after permission
                const devicesAfterPermission = await navigator.mediaDevices.enumerateDevices();
                const videoDevicesAfterPermission = devicesAfterPermission.filter(device => device.kind === 'videoinput');
                
                console.log("Video devices after permission:", videoDevicesAfterPermission.length);
                videoDevicesAfterPermission.forEach((device, i) => {
                    console.log(`Camera ${i+1}: ${device.label || 'unnamed device'}`);
                });
                
                return videoDevicesAfterPermission.length > 1;
            } catch (err) {
                console.error("Error getting camera permission for detection:", err);
                // If we can't get permission, assume there might be multiple cameras
                return true;
            }
        }
        
        return videoDevices.length > 1;
    } catch (error) {
        console.error('Error detecting cameras:', error);
        // If detection fails, assume there might be multiple cameras
        return true;
    }
}

// Clear the sentence
function clearSentence() {
    sentence = [];
    updateSentence();
}

// Socket.io event handlers
socket.on('connect', () => {
    console.log('Connected to server via WebSocket');
    isSocketConnected = true;
    statusElement.textContent = 'Status: WebSocket connected';
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    isSocketConnected = false;
    statusElement.textContent = 'Status: WebSocket disconnected';
});

socket.on('connected', (data) => {
    console.log('Server confirmation:', data);
});

socket.on('prediction_result', (data) => {
    statusElement.textContent = 'Status: Prediction received via WebSocket';
    processPrediction(data);
});

socket.on('prediction_error', (data) => {
    console.error('Prediction error:', data.error);
    statusElement.textContent = `Status: Prediction error - ${data.error}`;
});

// Event listeners
clearBtn.addEventListener('click', clearSentence);
fullscreenBtn.addEventListener('click', toggleFullscreen);
toggleCameraBtn.addEventListener('click', toggleCamera);
toggleLandmarksBtn.addEventListener('click', toggleLandmarksVisibility);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize canvas first
    initializeCanvas();
    
    // Show loading message
    statusElement.textContent = 'Status: Loading camera, please wait...';
    // Check for available cameras
    detectAvailableCameras().then(hasMultiple => {
        console.log("Multiple cameras detected:", hasMultiple);
        
        if (!hasMultiple) {
            toggleCameraBtn.disabled = true;
            toggleCameraBtn.title = "No additional cameras available";
        }
    });
    // Check if on mobile to set default landmarks visibility
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
        // If on mobile, hide landmarks by default for better performance
        showLandmarks = false;
        toggleLandmarksBtn.innerHTML = '<img src=".\\static\\icons\\eye-off.svg" alt="Show Landmarks" />';
        toggleLandmarksBtn.classList.add('landmarks-hidden');
    }
    
    // Initialize holistic and start camera
    initializeHolistic(() => {
        console.log('MediaPipe Holistic initialized, starting camera...');
        startCamera().catch(err => {
            console.error("Initial camera start failed:", err);
            // Try fallback if initial start fails
            startCameraWithFallback().catch(fallbackErr => {
                console.error("All camera start methods failed:", fallbackErr);
                statusElement.textContent = "Status: Could not start camera";
            });
        });
    });
    
    // Listen for fullscreen changes
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);
    
    // Listen for orientation changes
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            if (isFullscreen) {
                canvasElement.width = window.innerWidth;
                canvasElement.height = window.innerHeight;
            } else {
                const container = document.querySelector('.video-container');
                if (container) {
                    canvasElement.width = container.clientWidth;
                    canvasElement.height = container.clientHeight;
                }
            }
        }, 200);
    });
});