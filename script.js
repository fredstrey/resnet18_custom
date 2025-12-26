const API_URL = 'http://localhost:8000';

// Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const resultSection = document.getElementById('resultSection');
const previewImage = document.getElementById('previewImage');
const classResult = document.getElementById('classResult');
const confidenceResult = document.getElementById('confidenceResult');
const confidenceFill = document.getElementById('confidenceFill');
const loader = document.getElementById('loader');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
resetBtn.addEventListener('click', resetInterface);

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file processing
async function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload a valid image file (PNG, JPG, JPEG)');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    hideError();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Show result section and loader
    uploadArea.style.display = 'none';
    resultSection.classList.add('active');
    loader.classList.add('active');

    // Reset results
    classResult.textContent = '-';
    confidenceResult.textContent = '-';
    confidenceFill.style.width = '0%';

    // Send to API
    await classifyImage(file);
}

// Classify image using API
async function classifyImage(file) {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Classification failed');
        }

        const result = await response.json();
        displayResult(result);

    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}`);
        loader.classList.remove('active');
    }
}

// Display classification result
function displayResult(result) {
    loader.classList.remove('active');

    // Animate class name
    setTimeout(() => {
        classResult.textContent = result.class;
    }, 100);

    // Animate confidence
    setTimeout(() => {
        const confidencePercent = (result.confidence * 100).toFixed(2);
        confidenceResult.textContent = `${confidencePercent}%`;
        confidenceFill.style.width = `${confidencePercent}%`;
    }, 300);
}

// Reset interface
function resetInterface() {
    uploadArea.style.display = 'block';
    resultSection.classList.remove('active');
    fileInput.value = '';
    hideError();
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
}

// Hide error message
function hideError() {
    errorMessage.classList.remove('active');
}

// Check API health on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) {
            throw new Error('API is not responding');
        }
        console.log('✅ API is ready');
    } catch (error) {
        console.error('❌ API connection failed:', error);
        showError('Cannot connect to API. Please make sure the server is running on http://localhost:8000');
    }
}

// Initialize
checkAPIHealth();
