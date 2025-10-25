## 📋 README: Random Number Selection Prediction Challenge

This repository documents the solution for a data science competition focused on predicting an individual's chosen number, a task requiring the modeling of **human psychological bias** rather than pure statistical randomness.

The approach successfully utilized time-series and user-behavioral features to achieve a high-ranking solution (implied **3rd position**).

***

## 🚀 Project Overview

The challenge was to predict a single-digit number (0-9) likely to be selected by an individual, given a historical log of their selection attempts, timestamps, and time delays. The core strategy was to treat the selection as a **multiclass classification problem** and model the user's inherent, non-random biases and sequential patterns.

### Key Technologies

* **Python**
* **Pandas/NumPy:** Data manipulation and feature creation.
* **LightGBM (LGBMClassifier):** High-performance gradient boosting framework for classification.
* **Scikit-learn (StratifiedKFold, f1_score):** Cross-validation and evaluation metrics.
* **`datetime`/`pytz`:** Advanced temporal feature engineering.

***

## 🛠️ Data Preprocessing and Feature Engineering

The solution heavily relied on creating meaningful features from the raw, sparse activity log.

### 1. Data Cleaning & Imputation
* **Target Cleaning:** Non-numeric and missing values in the target **`number`** column were dropped, and the column was cast to integer.
* **Outlier Removal:** Rows with an extreme **`timedelay`** ($\ge 100$) were removed from the training set, indicating unreliable data points.
* **Timestamp Conversion:** Unix timestamps were converted to local timezone (Asia/Kolkata) datetime objects to facilitate temporal feature extraction.
* **Imputation:** Missing categorical and numerical values were imputed using learned parameters from the training set (e.g., median for `timedelay`, mode for `timestamp`).
* **Name Normalization:** User names appearing only once were grouped and labeled as **'anonymous'** to manage low-frequency noise.

### 2. Feature Generation

A robust set of features, derived from time-series and user-specific patterns, were created:

| Feature Category | Features Created | Description |
| :--- | :--- | :--- |
| **Temporal** | `hour_sin`, `hour_cos`, `day_of_week` | Circular and cyclical encoding of time to capture hourly and daily selection bias. |
| **Sequential/Lag** | **`lag1_number`** | The number previously selected by the user, crucial for modeling sequential human bias. |
| **User Statistics** | **`user_mean_delay`**, `user_std_delay` | Mean and standard deviation of the user's log-transformed `timedelay`, quantifying user hesitation/speed. |
| **Meta-Data** | `timedelay_log`, `name_freq`, `is_anonymous` | Log-transformation of time delay and frequency encoding of user names to capture impact on selection. |
| **Interaction** | `delay_x_hour` | Interaction between log-delay and hour to capture time-of-day effects on selection behavior. |

***

## 🧠 Model Training and Evaluation

### Model Architecture
The problem was framed as a 10-class multiclass classification task (numbers 0 through 9).

* **Model:** **LightGBM Classifier** (`lgb.LGBMClassifier`).
* **Objective:** `multiclass`, with `num_class=10`.
* **Handling Imbalance:** The model used `class_weight='balanced'` to prevent bias toward the most frequent numbers.

### Evaluation and Validation
* **Validation:** **5-Fold Stratified Cross-Validation** was performed using `StratifiedKFold` to ensure each fold maintained the distribution of the target number (0-9).
* **Metric:** **Macro F1-Score** was used as the primary metric, ensuring performance was evaluated equally across all 10 target numbers, not just the most common ones.

### Final Submission

The final model was trained on the complete, processed training set and used to generate predictions for the test set, creating the final `submission.csv` file.
