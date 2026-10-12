# Next-Word Prediction Using LSTM on Shakespeare's Hamlet

## Overview
This project trains an LSTM-based language model on the text of Shakespeare's *Hamlet*. It tokenizes the text, creates n-gram sequences, and learns to predict the next word from preceding words. The notebook then generates text by repeatedly predicting and appending the next word to a seed phrase.

## Dataset
The notebook uses the NLTK Gutenberg corpus file `shakespeare-hamlet.txt`, downloaded with `nltk.download('gutenberg')`.

## Workflow
1. Import NumPy, Matplotlib, NLTK, and TensorFlow/Keras components.
2. Download and load the Hamlet text from the Gutenberg corpus.
3. Convert text to lowercase and tokenize words.
4. Create n-gram sequences from each line.
5. Pad sequences and split them into input sequences (`X`) and next-word targets (`y`).
6. One-hot encode target words.
7. Train an Embedding + LSTM + Dropout + Softmax model.
8. Plot training loss and training/validation accuracy.
9. Generate continuations for seed phrases such as “to be or not” and “the king is”.

## Model Configuration
- Embedding dimension: 100
- LSTM units: 250
- Dropout: 0.4
- Output: Softmax over the tokenizer vocabulary
- Optimizer: Adam
- Loss: categorical cross-entropy
- Maximum epochs: 50
- Validation split: 20%
- Early stopping: monitors training loss, patience 5, restores best weights

## Requirements
Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Install the dependencies.
3. Run the cells from top to bottom.
4. The Gutenberg corpus is downloaded by NLTK during the notebook run; internet access may be needed.

## Notes
The notebook uses a word-level next-word prediction setup, not character-level generation. Results and generated phrases may vary with package versions and training behavior.
