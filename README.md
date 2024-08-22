# kaggle_Jigsaw-Unintended-Bias-in-Toxicity-Classification

## 1. Business Problem

**Source:** [Kaggle - Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

**Description:** [Competition Overview](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/description)

**Problem Statement:** Predict the toxicity of a user comment.

## 2. Machine Learning Problem Formulation

### 2.1 Data

**Source:** [Kaggle Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)

- **Train and Test Files:** CSV format.
- **Columns in Train Data:**
  - `comment_text`: The content to analyze for toxicity.
  - `target`: The toxicity score (0 to 1).
  - **Additional Subtypes:** e.g., `severe_toxicity`, `obscene`, `identity_attack`, etc.
  - **Identity Attributes:** e.g., `male`, `female`, `homosexual_gay_or_lesbian`, etc.
  - **Metadata:** e.g., `toxicity_annotator_count`, `article_id`, etc.

### 2.2 Example Data Points and Labels

**Example 1:**

- **Comment:** "I'm a white woman in my late 60's and believe me, they are not too crazy about me either!!"
- **Toxicity Labels:** All 0.0
- **Identity Labels:** `female`: 1.0, `white`: 1.0

**Example 2:**

- **Comment:** "Why would you assume that the nurses in this story were women?"
- **Toxicity Labels:** All 0.0
- **Identity Labels:** `female`: 0.8

### 2.3 Type of Machine Learning Problem

- **Problem Type:** Regression (0 to 1 toxicity score). Can be treated as a binary classification using a 0.5 threshold.

### 2.4 Performance Metric

- **Primary Metric:** ROC AUC based on a threshold of 0.5 for toxicity.
- **Training Metric:** Mean Squared Error (MSE)

### 2.5 Machine Learning Objectives and Constraints

- **Objective:** Accurately predict the toxicity of user comments.
- **Constraints:**
  - The model should be efficient in making predictions.
  - Interpretability of the model is not required.
