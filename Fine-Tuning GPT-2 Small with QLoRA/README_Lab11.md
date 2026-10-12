# GPT-2 Fine-Tuning: Baseline vs QLoRA

## Suggested Project Titles
1. **Comparative Study of GPT-2 Fine-Tuning and QLoRA on IMDb** (recommended)
2. Efficient GPT-2 Adaptation Using QLoRA
3. Parameter-Efficient Fine-Tuning of GPT-2 for Text Generation
4. Baseline Fine-Tuning vs 4-Bit QLoRA on the IMDb Dataset
5. Memory-Efficient Language Model Fine-Tuning Using LoRA
6. GPT-2 Text Generation with Quantization and LoRA
7. Comparing Full Fine-Tuning and Parameter-Efficient Fine-Tuning
8. QLoRA-Based Adaptation of GPT-2 for Movie Review Text
9. Efficient Transformer Fine-Tuning for Text Generation
10. GPT-2 Optimization Using 4-Bit Quantization and LoRA

## Overview
This notebook compares standard fine-tuning of GPT-2 with a parameter-efficient QLoRA setup on a subset of the IMDb movie-review dataset. It trains a baseline GPT-2 language model, then loads a 4-bit quantized GPT-2 model and applies LoRA adapters. It compares evaluation loss and perplexity, generates text from a prompt, and plots training losses.

## Dataset
The notebook uses Hugging Face Datasets:
- Dataset: `imdb`
- Training subset: 5,000 examples sampled from the train split
- Test subset: 1,000 examples sampled from the test split
- Validation: 10% split from the 5,000-example training subset
- Text max length: 128 tokens

The selected test subset is created but is not used directly in the training cells shown.

## Models and Configuration
### Baseline
- Model: `gpt2`
- Epochs: 1
- Per-device batch size: 4
- Logging every 100 steps
- Save every 500 steps
- `fp16=True`

### QLoRA
- Base model: `gpt2`
- Quantization: 4-bit (`load_in_4bit=True`)
- LoRA rank (`r`): 8
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target module: `c_attn`
- Task type: causal language modeling

## Workflow
1. Load and sample the IMDb dataset.
2. Tokenize text with the GPT-2 tokenizer and pad/truncate to 128 tokens.
3. Fine-tune a baseline GPT-2 model.
4. Load a 4-bit quantized GPT-2 model and attach LoRA adapters.
5. Train the QLoRA model.
6. Compare evaluation loss and perplexity.
7. Generate text from the prompt `The movie was`.
8. Plot baseline and QLoRA training-loss curves.

## Requirements
Install packages using:

```bash
pip install -r requirements_Lab11.txt
```

## How to Run
1. Open the notebook in a compatible GPU environment such as Google Colab.
2. Install dependencies from `requirements_Lab11.txt`.
3. Run cells in order. Hugging Face datasets and GPT-2 weights are downloaded as needed.
4. QLoRA's 4-bit quantization generally requires a compatible setup with CUDA and bitsandbytes; it may not run on CPU-only environments.

## Important Notes
- The notebook sets `fp16=True`, so the baseline training is intended for a compatible accelerator.
- The 4-bit model uses `BitsAndBytesConfig` and `device_map="auto"`; platform compatibility matters.
- The notebook creates `test_data`, but the shown evaluation uses `trainer.evaluate()` on the validation set by default.
- Package versions below are pinned suggestions and were not verified from the original notebook environment.
