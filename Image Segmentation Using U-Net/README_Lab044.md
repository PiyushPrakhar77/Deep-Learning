# Oxford-IIIT Pet Image Segmentation Using U-Net

## Suggested Project Titles
1. **Pet Image Segmentation Using U-Net** (recommended)
2. Oxford-IIIT Pet Segmentation with Deep Learning
3. Semantic Segmentation of Pet Images Using CNN
4. U-Net-Based Animal Image Segmentation
5. Pixel-Level Pet Segmentation Using TensorFlow
6. Deep Learning for Pet Foreground Segmentation

## Overview
This notebook builds a U-Net convolutional neural network to perform pixel-level segmentation on images from the Oxford-IIIT Pet dataset. The model predicts a segmentation class for each pixel rather than assigning one label to the whole image.

## Dataset
The notebook loads `oxford_iiit_pet:4.0.0` using TensorFlow Datasets (TFDS).
- Dataset: Oxford-IIIT Pet
- Inputs: RGB pet images
- Targets: segmentation masks
- Images and masks are resized to 256 × 256
- Batch size: 8

## Workflow
1. Load the Oxford-IIIT Pet dataset.
2. Extract each image and its segmentation mask.
3. Resize images and masks to 256 × 256.
4. Normalize image pixels to the range 0–1.
5. Adjust mask labels to start at 0.
6. Build a U-Net encoder-decoder architecture with skip connections.
7. Train for up to 10 epochs using Adam and sparse categorical cross-entropy.
8. Evaluate predictions and visualize input images, ground-truth masks, and predicted masks.
9. Plot training/validation loss and accuracy.
10. Create a pixel-wise confusion matrix.

## Model
- Architecture: U-Net
- Encoder/decoder convolution filters: 64, 128, 256, 512, 1024
- Output: 3 segmentation classes (default `num_classes=3`)
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Metric: Pixel accuracy
- Epochs: 10
- Batch size: 8
- Image size: 256 × 256

## Requirements
Python packages are listed in `requirements.txt`. The notebook downloads the dataset through TFDS, so internet access may be required on the first run.

## How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the cells in order.
4. If running outside Colab, ensure the notebook does not rely on the Colab-specific path `/content/heart.png`; that separate example cell expects a local image file.

## Important Implementation Note
The notebook contains an early example cell that loads `/content/heart.png` and creates a two-class model, while the main Oxford-IIIT Pet segmentation workflow uses a three-class U-Net. Run the main dataset/model cells in order and review that example cell separately if needed. The `mean_iou` function in the notebook currently computes pixel accuracy-like agreement, not the standard intersection-over-union metric.

## Results
Training plots, sample segmentation visualizations, and a pixel-wise confusion matrix are produced when the notebook is run. No numerical performance result is stated here because it depends on executing the notebook.
