# Stock Market Analysis and Prediction Project
Overview
This project aims to analyze historical stock market data, predict future stock prices using machine learning models, and perform sentiment analysis on news headlines and social media posts. Additionally, it leverages Ollama, a powerful language model, to interpret stock charts based on technical analysis. The insights generated from this project can help investors, traders, and financial analysts make informed decisions.

# Features
1. Portfolio Analysis
Build a portfolio by selecting multiple stocks or ETFs.

Visualize the performance of the portfolio over time.

Track percentage changes, moving averages, and other technical indicators.

2. Price Prediction
Predict future stock prices using machine learning models.

Supports models like LSTM, ARIMA, and Linear Regression.

3. Sentiment Analysis
Analyze sentiment from news headlines and social media posts related to selected stocks.

Uses a few-shot learning model to classify sentiment as positive, negative, or neutral.

4. Technical Analysis with Ollama
Generate stock charts with technical indicators like SMA (Simple Moving Average) and EMA (Exponential Moving Average).

Use Ollama (LLaMA 3.2 Vision) to interpret charts and provide actionable insights, such as buy/hold/sell recommendations.

5. Portfolio Calculator
Simulate investments by allocating amounts to selected stocks.

Track the growth of investments over time and set financial goals.

6. Real-Time Stock Prices
Display real-time stock prices and percentage changes for selected stocks.

## Table of Contents

1. [Installation](#Installation).

2. [Usage](#Usage).

3. [Model Training](#Model-Training).

4. [Output](#Example-Outputs).
   

## Installation
Prerequisites
Python 3.8 or higher

Streamlit for the web interface

Ollama for technical analysis interpretation

Required Python libraries (see requirements.txt)

Steps
Clone the Repository:
```
git clone https://github.com/yourusername/stock-market-analysis-prediction.git
cd stock-market-analysis-prediction
```
Install Dependencies:
```
pip install -r requirements.txt
```
Set Up Ollama:

Download and install Ollama from the official Ollama GitHub repository.

Ensure Ollama is running locally or on a server.

Run the Application:
```
streamlit run app.py
```

## Installing and Running Ollama on Your Local Machine
Step 1: Download and Install Ollama

Go to the official Ollama page.

Download the Installer:

Download the appropriate installer for your operating system (Windows, macOS, or Linux).

Install Ollama:

Step 2: Run Ollama Locally
Start the Ollama Server:

After installation, start the Ollama server by running the following command in your terminal:
```
ollama run llama3.2-vision
```
This will start the Ollama server on your local machine.

Verify the Installation:

Open a new terminal window and run the following command to verify that Ollama is running:
```
ollama list
```
If the installation is successful, you should see a list of available models.

## Usage
1. Portfolio Builder
   Select stocks or ETFs from the dropdown menu.

   View real-time prices and percentage changes for selected stocks.

   Analyze portfolio performance using interactive charts.

2. Sentiment Analysis
   Enter a news headline or fetch news related to selected stocks.

   View sentiment analysis results (positive, negative, or neutral).

3. Technical Analysis with Ollama
   Generate stock charts with technical indicators (e.g., SMA, EMA).

   Click "Run AI Analysis" to get buy/hold/sell recommendations from Ollama.

4. Portfolio Calculator
   Allocate investment amounts to selected stocks.

Set a financial goal and track progress over time.

Code Walkthrough
Key Functions
Fetching Stock Data:
```
def fetch_stock_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    return data
```
Fetches historical stock data using Yahoo Finance (yfinance).

Adding Technical Indicators:
```
def add_technical_indicators(data):
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data
```
Adds SMA and EMA to the stock data for technical analysis.

Sentiment Analysis:
```
def get_sentiment_analysis(tickers, start_date, end_date):
    # Fetch news headlines and classify sentiment using a few-shot model
    return sentiment_data
```
Uses a pre-trained few-shot learning model to classify sentiment.

Ollama Integration:
```
def analyze_chart_with_ollama(ticker, chart_data):
    # Generate chart and send to Ollama for analysis
    response = ollama.chat(model='llama3.2-vision', messages=messages)
    return response["message"]["content"]
```
Sends stock charts to Ollama for interpretation and recommendations.

Few-Shot Learning for Sentiment Analysis
Overview
The few-shot learning model is used to classify sentiment from news headlines and social media posts. It is particularly useful when labeled data is scarce, as it can generalize well with only a few examples.

## Model Training
The few-shot learning model is trained on a small dataset of labeled news headlines. Here's how the training process works:

Prepare the Dataset:

Collect a small dataset of news headlines labeled as positive, negative, or neutral.

Example dataset format:


Title, Sentiment
"Apple stock hits record high after earnings report", Positive
"Tesla faces production delays", Negative
"Microsoft announces new AI tools", Neutral
Train the Model:

Use a pre-trained language model (e.g., GPT, BERT) and fine-tune it on the labeled dataset.

Example code for training:

from transformers import pipeline

# Load a pre-trained model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["Positive", "Negative", "Neutral"]

# Train the model on a few examples
training_data = [
    {"text": "Apple stock hits record high after earnings report", "label": "Positive"},
    {"text": "Tesla faces production delays", "label": "Negative"},
    {"text": "Microsoft announces new AI tools", "label": "Neutral"},
]

for example in training_data:
    classifier(example["text"], candidate_labels=candidate_labels)
Save the Model:

Save the trained model for future use:

import joblib
joblib.dump(classifier, 'few_shot_model.pkl')
Load and Use the Model:

Load the trained model and use it for sentiment analysis:

loaded_few_shot_clf = joblib.load('few_shot_model.pkl')
sentiment = loaded_few_shot_clf("Apple stock hits record high after earnings report", candidate_labels=candidate_labels)
print(sentiment)


## Example Outputs
1. Portfolio Performance Chart
   Portfolio Performance

2. Sentiment Analysis Table
Ticker	Date	Title	Sentiment
AAPL	2023-10-01	Apple stock hits record high after earnings	Positive
TSLA	2023-10-02	Tesla faces production delays	Negative
3. Ollama Analysis

**AI Analysis Results:**
Recommendation: Buy
Reasoning: The stock chart shows a strong uptrend with the price consistently above the 20-day SMA and EMA. The RSI is below 70, indicating no overbought conditions. This suggests a good entry point for long-term investors.


