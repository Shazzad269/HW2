# HW2 for extra credit
# Extra Credit: MSFT Stock Price Forecasting using Residual CNN

This repository contains the implementation of the Extra Credit task for forecasting the stock price of Microsoft (MSFT) using a Residual 1D Convolutional Neural Network. The model is trained using PyTorch Lightning on historical stock data of MSFT and 19 other related companies.

## Objective

The goal is to predict the closing price of MSFT for the next day using the closing prices of 20 selected stocks from the previous 25 days. A deep CNN architecture with residual connections is used for improved performance. The final model achieves a test MSE of approximately 32.

## Dataset

- Historical daily closing prices are collected using the `yfinance` API.
- 20 stocks are used as input features, including MSFT.
- Each input sample consists of a 25-day window of prices, shaped as `[20, 25]`.
- The target is the MSFT closing price on the day following the 25-day input window.
- All features and the target are normalized using z-score normalization.

## Data Splitting

The dataset is split into three parts:

- 70% training
- 15% validation
- 15% testing

Splitting is done using PyTorch’s `random_split` with a fixed seed for reproducibility.

## Model Architecture

The model is built using PyTorch Lightning and consists of:

- 1D Convolutional layers:
  - Conv1d(20 → 32, kernel size=5)
  - Conv1d(32 → 64, kernel size=3)
  - Conv1d(64 → 32, kernel size=1)
- Batch Normalization after each convolution
- GELU activation functions
- Residual connection from the first to the third convolutional block
- Adaptive average pooling
- Fully connected layer to predict the normalized MSFT price
- Dropout for regularization

## Training Configuration

- Optimizer: AdamW
- Loss Function: Mean Squared Error (MSELoss)
- Learning Rate: 1e-4
- Batch Size: 32
- Epochs: 400
- Learning Rate Scheduler: StepLR with step size 50 and gamma 0.7

## Evaluation

- The model is evaluated on the test set using denormalized predictions and targets.
- The final test MSE is approximately **32**, calculated in the original price scale.

## How to Run

1. Install required packages:

2. Run the training and evaluation script:

3. The best model will be saved to:




