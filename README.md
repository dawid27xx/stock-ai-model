# stock-ai-model

## Description:

This program takes 20+ years of data from https://www.alphavantage.co/ and trains an AI model. Currently, the data is on a weekly interval and evaluates if in the next week, the stock will go up or down. 

## Dependencies:

- Numpy
- Pandas
- Requests
- PyTorch
- SkLearn

## Usage:

Create virtual environment, install dependencies, and finally enter the env and run the program.

When ran, the user will be asked if to use mock or real data. Mock data is based on the IBM stock. If the user chooses real data, they will then be asked to input a ticker simple (example tickers: AAPL, MSFT, NVDA). Once that is inputed, the program will then get the data, process it, train the model and finally evaluate for its accuracy.

## Sample Output (Mock Data):
<img width="383" alt="Screenshot 2024-09-01 at 11 44 51" src="https://github.com/user-attachments/assets/83dd084f-35e3-4f94-8bb2-76595871b8c3">

## Indicators Used:

- Simple Moving Average
- Relative Strength Index
- On Balance Volume
- Previous Close
- Previous Weekly Change

## Areas for Improvement: 

- Let the user choose the interval
- Keep track of user's portfolio
- Experiment with more technical indicators to improve the model
- Include Sentimental Analysis

## Results

- Nasdaq: loss: 0.64, accuracy: 70.1% 
- IBM: loss: 0.65, accuracy: 66.8%
- Nvidia: loss: 0.68, accuracyL 58.7%
- Google: loss: 0.67, accuracy: 60.8%

Evaluation of results: The base result for a binary classification model is ~50%, and the model exceeds that.
The model is best with the Nasdaq due to its lower volatility as opposed to individual stocks. IBM yields a high accuracy due to its long periods of fixed pricing. The model is not as accurate in individual stocks with lows going down to 58.7% in volatile stocks such as Nvidia.



