# Stock Price Movement Prediction using Machine Learning

## 📈 Project Overview

This project aims to predict the daily directional movement (up or down) of a stock's price using historical price and volume data. We leverage a combination of advanced feature engineering, machine learning (XGBoost), and deep learning (LSTM) to build and evaluate predictive models. The workflow is designed to be a robust, end-to-end pipeline for financial time series classification.

The primary model, a **Tuned XGBoost Classifier**, achieved a **ROC-AUC of 0.5312** on the test set, demonstrating a slight predictive edge in the highly complex and efficient financial market.

---

## 🛠️ Technologies & Libraries

- **Programming Language:** Python 3.9+
- **Core Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Data Acquisition:** `yfinance`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Deep Learning:** `tensorflow` (Keras)
- **Hyperparameter Tuning:** `optuna`
- **Environment:** Jupyter Notebook

---


---

## ⚙️ Methodology & Workflow

The project follows a structured, multi-stage process:

### 1. Data Acquisition & Feature Engineering
- **Data Source:** Fetched over 10 years of daily stock data for Apple Inc. (`AAPL`) using the `yfinance` library.
- **Target Definition:** The problem was framed as a binary classification task: predict `1` if the next day's close price is higher, and `0` otherwise.
- **Feature Creation:** Engineered a comprehensive set of **30+ technical indicators and statistical features**, including:
  - **Trend:** Simple Moving Averages (SMA), Exponential Moving Averages (EMA).
  - **Momentum:** Rate of Change (ROC), Relative Strength Index (RSI), Williams %R.
  - **Volatility:** Bollinger Bands (position within the bands).
  - **Lag Features:** Past values of price and volume to capture autocorrelation.

### 2. Model Training: XGBoost
- **Algorithm:** Utilized XGBoost (eXtreme Gradient Boosting), a powerful and efficient tree-based ensemble algorithm known for its high performance in competitions.
- **Validation Strategy:** Implemented a chronological train-test split (80/20) to ensure the model was validated on unseen future data, simulating real-world forecasting.

### 3. Hyperparameter Optimization with Optuna
- **Automated Tuning:** Employed the `Optuna` framework to perform Bayesian optimization, automatically searching for the best combination of hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`).
- **Objective:** The tuning process was optimized to maximize the **ROC-AUC score**, which is a robust metric for classification performance.

### 4. Deep Learning Comparison: LSTM
- **Sequential Modeling:** Built a Long Short-Term Memory (LSTM) network, a type of Recurrent Neural Network (RNN) specifically designed for sequence data, as a deep learning benchmark.
- **Data Preparation:** Transformed the feature set into 3D sequences `(samples, time_steps, features)` required for LSTM input.
- **Training:** The model was trained with `EarlyStopping` to prevent overfitting.

### 5. Evaluation & Interpretation
- **Quantitative Analysis:** Compared models using a suite of metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Qualitative Analysis:**
  - **Confusion Matrix:** Visualized the model's true/false positive and negative predictions.
  - **Feature Importance:** Analyzed the XGBoost model to identify the most influential technical indicators.

---

## 📊 Key Results

| Model             | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| **Tuned XGBoost** | **51.86%** | **55.13%** | 57.62% | 56.34%   | **0.5312**  |
| LSTM              | 52.87%   | 53.50%    | 91.52% | 67.53%   | 0.4872  |

- The **Tuned XGBoost** model was selected as the final model due to its superior ROC-AUC score, indicating a more reliable ability to distinguish between "up" and "down" days compared to the LSTM.
- The most important features identified were the **Middle Bollinger Band**, **lagged closing prices**, and **volatility indicators** like Williams %R.

---


---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
