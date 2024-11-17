document.getElementById('analyzeButton').addEventListener('click', async () => {
    const imageInput = document.getElementById('wasteImage');
    
    if (imageInput.files.length > 0) {
        const imageFile = imageInput.files[0];
        
        // Load and preprocess the image here
        const img = await loadImage(imageFile);
        
        // Call your AI model here
        const result = await analyzeWaste(img);
        
        // Display results
        document.getElementById('output').innerText = result ? "This is recyclable!" : "This is not recyclable.";
    } else {
        alert("Please upload an image.");
    }
});
async function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => resolve(img); // Resolve with loaded image
            img.onerror = (error) => reject(error); // Handle errors
        };
        
        reader.onerror = (error) => reject(error);
        
        reader.readAsDataURL(file); // Read file as Data URL
    });
}
async function analyzeWaste(img) {
    // Placeholder for AI model logic; replace with actual model call.
    // For example: 
    // const response = await fetch('/api/analyze', { method: 'POST', body: img });
    // return response.json();

    // Simulated analysis logic for demonstration purposes:
    return Math.random() > 0.5; // Randomly returns true or false for demo purposes.
}
import * as tf from './node_modules/@tensorflow/tfjs/dist/tf.min.js';

// Define the CNN model
function createModel() {
    const model = tf.sequential();

    // First convolutional layer
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: [3, 3],
        activation: 'relu',
        inputShape: [150, 150, 3] // Input shape for images (height, width, channels)
    }));

    // First max pooling layer
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));

    // Second convolutional layer
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: [3, 3],
        activation: 'relu'
    }));

    // Second max pooling layer
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));

    // Third convolutional layer
    model.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: [3, 3],
        activation: 'relu'
    }));

    // Third max pooling layer
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));

    // Flattening the output from the convolutional layers
    model.add(tf.layers.flatten());

    // Fully connected layer with ReLU activation
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }));

    // Output layer with sigmoid activation for binary classification
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid' 
    }));

    return model;
}

// Create the CNN model instance
const model = createModel();

// Optionally compile the model (this step is necessary before training)
model.compile({
   optimizer: tf.train.adam(), // Adam optimizer
   loss: 'binaryCrossentropy',   // Loss function for binary classification
   metrics: ['accuracy']         // Metrics to track during training and evaluation
});

// Placeholder function to analyze waste using the created AI model.
   // Logic to run the AI model on the image goes here

   const tensorImage = tf.browser.fromPixels(image).resizeNearestNeighbor([150, 150]).toFloat().expandDims(0);
   
   const prediction = await model.predict(tensorImage).data();
   
function isRecyclable(prediction) {
    if (Array.isArray(prediction) && prediction.length > 0) {
        return prediction[0] > 0.5; // Returns true if recyclable based on threshold of 0.5
    } else {
        return false; // Default case if prediction is not valid
    }
}
