# Toxic Comment Classification Using BiLSTM Model

## 1. Business Problem

**Source:** [Kaggle - Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

**Data:** https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

**Problem Statement:** Predict the toxicity of user comments while considering potential biases in the data.

## 2. Machine Learning Problem Formulation

### 2.1 Data

**Source:** Kaggle Dataset

- **Train and Test Files:** CSV format
- **Key Columns:**
  - `comment_text`: Text data to analyze.
  - `target`: Toxicity score (0 to 1).
  - **Additional Labels:** Includes severe toxicity, obscene, identity attack, etc.
  - **Identity Attributes:** E.g., male, female, homosexual_gay_or_lesbian, etc.
  - **Metadata:** Includes annotator counts, article ID, etc.

### 2.2 Example Data Points and Labels

- **Example 1:**
  - **Comment:** "I'm a white woman in my late 60's and believe me, they are not too crazy about me either!!"
  - **Toxicity Labels:** All 0.0
  - **Identity Labels:** Female: 1.0, White: 1.0

- **Example 2:**
  - **Comment:** "Why would you assume that the nurses in this story were women?"
  - **Toxicity Labels:** All 0.0
  - **Identity Labels:** Female: 0.8

### 2.3 Type of Machine Learning Problem

- **Problem Type:** Regression (toxicity score from 0 to 1). Can also be treated as a binary classification using a 0.5 threshold.

### 2.4 Performance Metric

- **Primary Metric:** ROC AUC based on a threshold of 0.5 for toxicity.
- **Training Metric:** Mean Squared Error (MSE).

### 2.5 Machine Learning Objectives and Constraints

- **Objective:** Accurately predict the toxicity of user comments.
- **Constraints:**
  - The model should be efficient in making predictions.
  - Interpretability is not a priority.

## 3. Model Overview

### 3.1 Architecture

- **Embedding Layer:** Uses pre-trained GloVe and FastText embeddings.
- **BiLSTM Layers:** Two bidirectional LSTM layers to capture the sequence context.
- **Dense Layers:** Fully connected layers for classification.
- **Spatial Dropout:** Applied to the embedding layer to reduce overfitting.

### 3.2 Training Strategy

- **Loss Function:** Binary Cross-Entropy with Logits Loss (`nn.BCEWithLogitsLoss`).
- **Optimizer:** Adam optimizer with learning rate scheduling.
- **Ensemble Learning:** Multiple models are trained, and predictions are averaged to improve performance.

## 4. Execution

### 4.1 Running the Code

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data:**
   - Modify paths to point to your training and test datasets.

3. **Train the Model:**
   - Run the script to train the model and generate predictions.

4. **Generate Predictions:**
   - The script outputs predictions to a CSV file (`result.csv`).

## 5. Dependencies

- Python 3.9+
- TensorFlow
- PyTorch
- NumPy, Pandas, and other essential libraries.

## 6. License

This project is licensed under the APACHE2.0 License. See the LICENSE file for details.
