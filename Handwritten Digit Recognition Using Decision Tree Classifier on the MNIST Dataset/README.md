# MNIST Digit Recognition

## Overview
This projrct uses a Decision Tree Classifier to recognize handwritten digits (0–9) from the MNIST dataset.

## Dataset
- Dataset: `mnist_784` loaded with Scikit-learn's `fetch_openml()`
- Images: 70,000
- Image size: 28 × 28 pixels
- Features per image: 784
- Classes: digits 0–9

## Workflow
pip install -r requirements.txt :- use this to install all requirements 
1. Load the MNIST dataset.
2. Inspect the dataset dimensions and labels.
3. Visualize sample handwritten digits.
4. Represent each image using 784 pixel features.
5. Normalize pixel values to the range 0–1.
6. Split the data into 80% training and 20% testing sets.
7. Train a Decision Tree Classifier using the entropy criterion.
8. Predict labels for the test set.
9. Evaluate the model using accuracy.

## Tools and Libraries
- Python
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## Model Settings
- Classifier: `DecisionTreeClassifier`
- Criterion: `entropy`
- Random state: `42`
- Test size: `0.2`

## Result
The uploaded lab output reports an accuracy of approximately **88.02%**.

## How to Run
1. Download `code.ipynb`.
2. Open [Google Colab](https://colab.research.google.com/).
3. Choose **File → Upload notebook** and upload the notebook.
4. Run the cells from top to bottom. The dataset may need to be downloaded from OpenML.

## Note
Some outputs are embedded from the original PDF. Run the notebook cells to reproduce the computation and generate any outputs not already saved.
