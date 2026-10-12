# Adversarial Robustness of CNNs on CIFAR-10

## Suggested Project Titles
1. **Evaluating CNN Robustness Against Adversarial Attacks on CIFAR-10** (recommended)
2. CIFAR-10 Image Classification Under Adversarial Attacks
3. Adversarial Image Attacks: FGSM, BIM, and PGD
4. Robustness Analysis of Convolutional Neural Networks
5. Adversarial Examples and CNN Classification on CIFAR-10
6. Comparing FGSM, BIM, and PGD Attacks on Image Classifiers
7. Deep Learning Security: Adversarial Attacks on CIFAR-10
8. Evaluating Image Classifier Performance on Clean and Perturbed Images
9. Adversarial Robustness Testing Using TensorFlow
10. Impact of Gradient-Based Attacks on CNN Image Classification

## Overview
This project trains a Convolutional Neural Network (CNN) on CIFAR-10 and evaluates its performance on both clean images and adversarially perturbed images. It implements three gradient-based attacks: Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), and Projected Gradient Descent (PGD).

## Dataset
The notebook loads CIFAR-10 through `tf.keras.datasets.cifar10`.
- 50,000 training images and 10,000 test images
- RGB images of size 32 × 32
- 10 image classes
- Pixel values are normalized to [0, 1]
- Labels are one-hot encoded

## CNN Architecture
- Conv2D: 32 filters, 3 × 3 kernel, ReLU, same padding
- MaxPooling2D: 2 × 2
- Conv2D: 64 filters, 5 × 5 kernel, ReLU, same padding
- MaxPooling2D: 2 × 2
- Conv2D: 128 filters, 7 × 7 kernel, ReLU, same padding
- MaxPooling2D: 2 × 2
- Flatten
- Dense: 128 units, ReLU
- Dropout: 0.3
- Output: 10 units, Softmax

## Training Configuration
- Optimizer: Adam
- Loss: categorical cross-entropy
- Batch size: 512
- Maximum epochs: 100
- Validation split: 20%
- Early stopping: monitors validation loss, patience 5, restores best weights
- Model checkpoint: saves the best model by validation accuracy to `best_model.keras`

## Adversarial Attacks
- **FGSM:** one gradient-sign step, epsilon 0.03 in the example evaluation.
- **BIM:** iterative FGSM steps, default epsilon 0.03, step size 0.005, 10 iterations.
- **PGD:** random initialization within the epsilon range followed by iterative projected steps; default epsilon 0.03, step size 0.005, 20 iterations.

The notebook evaluates clean and attacked test images using accuracy, loss, classification reports, confusion matrices, and example visualizations.

## Requirements
Install packages with:

```bash
pip install -r requirements.txt
```

## How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies from `requirements.txt`.
3. Run all cells in order. CIFAR-10 is downloaded automatically by Keras.
4. The notebook saves the best checkpoint as `best_model.keras`.

## Notes
The README does not state a fixed accuracy because the model must be trained in the runtime. The listed package versions are pinned recommendations, not confirmed versions from the notebook's original environment.
