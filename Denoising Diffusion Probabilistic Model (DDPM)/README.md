# Pet Image Generation with a DDPM

## Overview
This notebook builds a simple Denoising Diffusion Probabilistic Model (DDPM) in PyTorch to generate pet images. It uses a custom U-Net to predict the noise added to images at different diffusion timesteps.

## Dataset
- **Dataset:** Oxford-IIIT Pet image collection (images archive)
- **Source:** https://www.robots.ox.ac.uk/~vgg/data/pets/
- Images are downloaded by the notebook and extracted into `/content/pet_dataset`.
- Images are converted to RGB, resized to `64 × 64`, converted to tensors, and normalized to `[-1, 1]`.
- The notebook uses the images only; pet labels are not used.

## Model and configuration
- **Framework:** PyTorch
- **Architecture:** Custom U-Net with sinusoidal timestep embeddings
- **Diffusion steps (`T`):** 1000
- **Noise schedule:** Linear beta schedule from `0.0001` to `0.02`
- **Training objective:** Mean Squared Error (MSE) between actual noise and predicted noise
- **Optimizer:** Adam
- **Learning rate:** `0.0002`
- **Epochs:** 40
- **Batch size:** 32
- **Train/test split:** 80% / 20%
- **Random seed:** 42

## Notebook workflow
1. Import libraries and select CUDA when available.
2. Download and inspect pet images.
3. Preprocess images and create train/test data loaders.
4. Define the forward diffusion noise schedule.
5. Add noise to images and visualize selected timesteps.
6. Build and check the timestep-conditioned U-Net.
7. Train the model to predict noise.
8. Plot training loss.
9. Run reverse diffusion to generate pet images and visualize intermediate denoising.
10. Compare generated samples with real images and save outputs.

## Run the notebook
1. Open `model.ipynb` in Google Colab or a Jupyter environment.
2. Install the packages listed in `requirements.txt`.
3. Run the notebook cells in order.
4. Ensure internet access is available for downloading the dataset.
5. A CUDA-enabled GPU is recommended. Training and generating images with 1000 reverse-diffusion steps can be slow on CPU.

## Outputs
The notebook saves:
- Model weights: `/content/ddpm_pet_model.pth`
- Generated image grid: `/content/ddpm_pet_outputs/generated_pet_images.png`

These paths are designed for Google Colab. Change them if running locally.

## Notes and limitations
- The notebook trains a small educational DDPM; image quality depends on training time and available hardware.
- The held-out test split is used for visualization, not for a quantitative image-generation metric.
- The dependency versions in `requirements.txt` are recommended pins and were not confirmed from the original runtime.
