# English-to-French Neural Machine Translation with Attention

## Overview
This notebook builds sequence-to-sequence models to translate English sentences into French. It uses an LSTM encoder and decoder and compares three attention approaches: additive attention, Luong-style dot-product attention, and multi-head cross-attention.

## Dataset
The notebook loads `eng_-french.csv` from `/content/` and renames its two columns to `english` and `french`. The dataset file must be provided separately at that location, or the path should be updated for your environment.

## Workflow
1. Load and clean English and French text.
2. Tokenize both languages with Keras Tokenizer.
3. Convert sentences to token sequences and post-pad them.
4. Split the data into training and test subsets (80/20, `random_state=42`).
5. Prepare decoder input and next-token target sequences.
6. Build and train an LSTM encoder–decoder with additive attention.
7. Build and train a Luong dot-product attention model.
8. Build and train a multi-head cross-attention model.
9. Plot attention weights for a sample.
10. Compare model predictions and attempt BLEU evaluation.

## Model Configuration
- Vocabulary size: 10,000
- Embedding dimension: 256
- LSTM latent dimension: 256
- Batch size: 64
- Epochs: 10 per model
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Attention variants: additive, Luong dot-product, and multi-head cross-attention (`num_heads=4`)

## Requirements
Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run
1. Place `eng_-french.csv` in the expected path or update the notebook's `pd.read_csv()` path.
2. Install dependencies.
3. Run the notebook cells in order.
4. Review attention visualizations, model comparisons, and BLEU results.

## Important Notes
- The notebook's BLEU function references `model_soft`, but the additive-attention model in the notebook is named `model`; this may require changing `model_soft` to `model`.
- Standard BLEU evaluation should use word-token lists. The notebook passes integer token IDs, so its BLEU scores may not be meaningful without adjusting the evaluation.
- The notebook defines attention weights in the additive-attention model, while Luong and cross-attention models are evaluated through their prediction outputs.
- Results depend on dataset content, training runtime, and package versions.
