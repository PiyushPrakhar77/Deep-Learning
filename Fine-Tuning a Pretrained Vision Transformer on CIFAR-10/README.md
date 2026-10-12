# CIFAR-10 Image Classification Using Vision Transformer (ViT)

## Overview
This notebook fine-tunes a pretrained Vision Transformer (ViT) for 10-class image classification on CIFAR-10. The pretrained checkpoint is `google/vit-base-patch16-224-in21k`, loaded through Hugging Face Transformers. Images are resized to 224 × 224 and normalized using ImageNet mean and standard deviation.

## Dataset
CIFAR-10 is downloaded automatically using `torchvision.datasets.CIFAR10`.
- Training images: 50,000
- Test images: 10,000
- Image size before transform: 32 × 32 RGB
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Workflow
1. Install and import the required libraries.
2. Load CIFAR-10 and apply resize, tensor conversion, and ImageNet normalization.
3. Create training and test DataLoaders with batch size 32.
4. Load the pretrained ViT checkpoint and adapt its classifier to 10 classes.
5. Train with CrossEntropyLoss and AdamW.
6. Evaluate test loss and accuracy after each epoch.
7. Print a classification report.
8. Plot loss and accuracy curves.
9. Visualize sample predictions alongside their true labels.

## Model and Training Configuration
- Model: `google/vit-base-patch16-224-in21k`
- Frameworks: PyTorch, Hugging Face Transformers, Torchvision
- Number of classes: 10
- Optimizer: AdamW
- Learning rate: 0.00005
- Loss: Cross-entropy
- Batch size: 32
- Epochs: 3
- Image resolution: 224 × 224

## Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run
1. Open the notebook in Google Colab or a compatible Python environment.
2. Install the packages listed in `requirements.txt`.
3. Run the notebook cells from top to bottom.
4. The CIFAR-10 dataset and pretrained ViT weights are downloaded automatically, so internet access is needed on the first run.

## Notes
- The notebook selects CUDA when available and otherwise runs on CPU. ViT fine-tuning can be slow on CPU.
- Package versions below are pinned recommendations, not versions verified from the original notebook runtime.
- Test accuracy is produced by running the notebook; no fixed accuracy is stated here.
