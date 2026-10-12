# Wine Quality Classification

## Overview
This code explores the white wine quality dataset and builds a neural network to classify wine quality categories. The notebook includes data exploration, visualization, preprocessing, model training, and evaluation.

## Dataset
The lab uses the wine quality CSV files:
- `winequality-white.csv`
- `winequality-red.csv`

The PDF notebook shown uses `winequality-white.csv`. The white wine dataset contains 4,898 rows and 12 columns, including 11 physicochemical features and the `quality` target.

## Workflow
1. Load the dataset into a Pandas DataFrame.
2. Inspect the first and last rows, data types, summary statistics, and missing values.
3. Visualize feature distributions with histograms.
4. Plot a correlation heatmap.
5. Separate the input features from the `quality` target.
6. Encode quality labels and convert them to categorical format.
7. Normalize input features using `MinMaxScaler`.
8. Split the data into training and testing sets (80/20).
9. Train a neural network using TensorFlow/Keras with Dense and Dropout layers.
10. Plot training and validation accuracy/loss.
11. Evaluate predictions using a confusion matrix and classification report.

## Technologies
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras
- Jupyter Notebook / Google Colab

## How to Run
1. Keep the CSV files in the notebook's working directory, or upload them to Colab.
2. Update the CSV path if needed. For example:
   `pd.read_csv("winequality-white.csv", sep=";")`
3. Open the notebook in Google Colab or Jupyter.
4. Run the cells from top to bottom.

## Notes
- The CSV files use a semicolon (`;`) as the delimiter.
- The PDF shows a seven-class output dimension for the encoded target.
- Training results may vary with library versions, random initialization, and runtime environment.
