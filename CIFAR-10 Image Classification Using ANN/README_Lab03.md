# CIFAR-10 Image Classification Using an Artificial Neural Network (ANN)

## Project Title
**CIFAR-10 Image Classification Using ANN**

## Overview
This project trains a fully connected Artificial Neural Network (ANN) to classify images from the CIFAR-10 dataset into 10 categories. The notebook uses TensorFlow/Keras to build, train, and evaluate the model.

## Dataset
The notebook loads CIFAR-10 directly using `tensorflow.keras.datasets.cifar10`.

- Training images: 50,000
- Test images: 10,000
- Image dimensions: 32 × 32 × 3 (RGB)
- Classes: 10

## Workflow
1. Load the CIFAR-10 dataset.
2. Normalize image pixel values to the range 0–1.
3. Flatten each image into a 3,072-feature vector.
4. One-hot encode the class labels.
5. Build a dense neural network with ReLU hidden layers, Dropout, and a Softmax output layer.
6. Compile the model using Adam, categorical cross-entropy, and accuracy.
7. Train with a validation split and EarlyStopping.
8. Evaluate test accuracy and loss.
9. Plot a confusion matrix and training/validation accuracy and loss.

## Model Configuration
- Hidden Dense layers: 1024, 512, and 256 units
- Hidden activation: ReLU
- Dropout: 0.4 after the first two Dense layers
- Output layer: 10 units with Softmax
- Optimizer: Adam
- Learning rate: 0.0005
- Loss: Categorical cross-entropy
- Batch size: 256
- Maximum epochs: 250
- Early stopping: monitors validation loss with patience 5 and restores best weights

## Libraries
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the notebook cells from top to bottom.
4. The CIFAR-10 dataset will be downloaded automatically by Keras on the first run, so an internet connection may be required.

## Notes
The notebook's exact test accuracy depends on training execution and runtime conditions. No final accuracy value is specified in this README because it should be taken from the output generated when the notebook is run.
