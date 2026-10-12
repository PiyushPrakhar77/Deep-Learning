# CIFAR-10 Image Classification Using CNN

## Overview
This project builds and evaluates a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories. It uses TensorFlow/Keras and includes model evaluation and visualization of learned feature maps.

## Dataset
The notebook loads CIFAR-10 using `tensorflow.keras.datasets.cifar10`.

- Training images: 50,000
- Test images: 10,000
- Image dimensions: 32 × 32 × 3 (RGB)
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Workflow
1. Load and normalize the image data to the range 0–1.
2. One-hot encode the class labels.
3. Build a CNN with convolutional, max-pooling, dense, and dropout layers.
4. Train with validation data and early stopping.
5. Plot training/validation accuracy and loss.
6. Evaluate test accuracy and generate a classification report and confusion matrix.
7. Visualize convolutional feature maps.
8. Display misclassified test images.

## Model Architecture
- Conv2D: 32 filters, 3 × 3 kernel, ReLU
- MaxPooling2D: 2 × 2
- Conv2D: 64 filters, 5 × 5 kernel, ReLU
- MaxPooling2D: 2 × 2
- Conv2D: 128 filters, 7 × 7 kernel, ReLU
- MaxPooling2D: 2 × 2
- Flatten
- Dense: 256 units, ReLU
- Dropout: 0.8
- Dense output: 10 units, Softmax

## Training Configuration
- Optimizer: Adam (default settings)
- Loss: categorical cross-entropy
- Batch size: 256
- Maximum epochs: 25
- Validation split: 10%
- Early stopping: monitors validation loss, patience 5, restores best weights

## Libraries
NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras.

## How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the cells from top to bottom.
4. Keras downloads CIFAR-10 automatically on first use; internet access may be required.

## Notes
Test accuracy and training time are generated when the notebook runs and may vary by environment. The notebook evaluates one CNN model; it does not compare multiple model architectures.
