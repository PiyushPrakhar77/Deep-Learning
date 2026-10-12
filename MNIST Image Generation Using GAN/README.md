# Fashion-MNIST Image Generation Using a GAN

## Overview
This project implements a Generative Adversarial Network (GAN) that learns to generate synthetic grayscale fashion-item images using the Fashion-MNIST dataset. The GAN consists of a generator that creates images from random noise and a discriminator that distinguishes real images from generated images.

## Dataset
The notebook loads Fashion-MNIST directly through `tf.keras.datasets.fashion_mnist`.

- Training images used: 60,000
- Image dimensions: 28 × 28 grayscale
- Pixel values are normalized to the range [-1, 1]
- The notebook uses the training images for GAN training; the test split is loaded but not used.

## Model Architecture
### Generator
- Latent noise vector size: 100
- Dense layer projecting to 7 × 7 × 128
- Batch normalization and LeakyReLU
- Transposed convolution layers upsampling to 28 × 28
- `tanh` output activation

### Discriminator
- Two convolutional layers with 64 and 128 filters
- LeakyReLU activations and Dropout (0.3)
- Flatten layer and a single logit output

## Training Configuration
- Loss: Binary cross-entropy with logits
- Optimizer: Adam
- Learning rate: 0.0001 for both networks
- Batch size: 256
- Epochs: 30
- Latent dimension: 100
- Generated image samples displayed every 5 epochs

## Workflow
1. Load Fashion-MNIST.
2. Reshape and normalize images.
3. Create a shuffled batched TensorFlow dataset.
4. Build the generator and discriminator.
5. Define GAN losses and optimizers.
6. Train both networks with `tf.GradientTape`.
7. Display generated images during training.
8. Plot generator and discriminator losses.

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Install the dependencies.
3. Run all cells from top to bottom.
4. The Fashion-MNIST dataset is downloaded by Keras automatically and may require internet access on the first run.

## Notes
This notebook demonstrates image generation, not classification. The generated samples and loss curves depend on training and runtime conditions. The dependency versions below are pinned recommendations and were not recovered from the notebook's original environment.
