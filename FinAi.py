import json
import numpy as np 
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
from datetime import date
from sklearn.preprocessing import StandardScaler


# free api key; 25 requests per day
APIKEY = '28U0VBI1N5ZOJOEU'

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, output_size)  
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


def main():    
    realData = input("Real Data or Mock? (r/m)").lower()
    if realData == "r":
        stock = input("Input Ticker: ").upper()
        print(f'Getting Market Data on {stock}...')
    else:
        print("Getting Market Data on IBM...")
        stock = "IBM"    
    
    stockData, SMAdata, RSIdata, OBVdata = getData(realData, stock)
    
    print("Processing Data...")
    processedData = processData(stockData, SMAdata, RSIdata, OBVdata)
    trainData, testData, inputSize = splitData(processedData)
    
    print("Training Model...")
    model, criterion = trainModel(trainData, inputSize)
    
    print("Evaluating Model...")
    print(evaluateModel(model, testData, criterion))
    
    
    
def getData(realData, stock):
    if realData == 'r':
        stockDataUrl = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={stock}&apikey={APIKEY}'
        SMAdataUrl = f'https://www.alphavantage.co/query?function=SMA&symbol={stock}&interval=weekly&time_period=10&series_type=open&apikey={APIKEY}'
        RSIdataUrl = f'https://www.alphavantage.co/query?function=RSI&symbol={stock}&interval=weekly&time_period=10&series_type=open&apikey={APIKEY}'
        OBVdataUrl = f'https://www.alphavantage.co/query?function=OBV&symbol={stock}&interval=weekly&apikey={APIKEY}'
    else:
        stockDataUrl = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=demo'
        SMAdataUrl = 'https://www.alphavantage.co/query?function=SMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
        RSIdataUrl = 'https://www.alphavantage.co/query?function=RSI&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
        OBVdataUrl = 'https://www.alphavantage.co/query?function=OBV&symbol=IBM&interval=weekly&apikey=demo'

    stockDataRequest = requests.get(stockDataUrl)
    SMAdataRequest = requests.get(SMAdataUrl)
    RSIdataRequest = requests.get(RSIdataUrl)
    OBVdataRequest = requests.get(OBVdataUrl)
    stockData = stockDataRequest.json()
    SMAdata = SMAdataRequest.json()
    RSIdata = RSIdataRequest.json()
    OBVdata = OBVdataRequest.json()
    
    return stockData, SMAdata, RSIdata, OBVdata


def processData(stockData, SMAdata, RSIdata, OBVdata):
    stockDataProcessed = processStockData(stockData)
    stockDataProcessed = combineData(stockDataProcessed, SMAdata, RSIdata, OBVdata)    
    return stockDataProcessed 
    
    
def processStockData(stockData):
    weekly_series = stockData["Weekly Time Series"]
    df = pd.DataFrame.from_dict(weekly_series, orient='index')
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    
    df = df.astype({
        'open': 'float',
        'close': 'float',
        'close': 'float',
    })
    
    df['weekChange'] = (df['close'] - df['open']) / df['open'] * 100
    df['weekChangeBinary'] = (df['weekChange'] > 0).astype(int) 
    
    df['prev_close'] = df['close'].shift(-1)
    df['prev_weekChange'] = df['weekChange'].shift(-1)
        
    features = ['prev_close', 'prev_weekChange']
    df = df[features + ['weekChange', 'weekChangeBinary']] 
    
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    
    return df

def combineData(stockData, SMAdata, RSIdata, OBVdata):
    SMAdata = SMAdata['Technical Analysis: SMA']
    dfSMA = pd.DataFrame.from_dict(SMAdata, orient='index')
    dfSMA.columns = ['SMA']
    dfSMA.index = pd.to_datetime(dfSMA.index)

    RSIdata = RSIdata['Technical Analysis: RSI']
    dfRSI = pd.DataFrame.from_dict(RSIdata, orient='index')
    dfRSI.columns = ['RSI']
    dfRSI.index = pd.to_datetime(dfRSI.index)

    OBVdata = OBVdata['Technical Analysis: OBV']
    dfOBV = pd.DataFrame.from_dict(OBVdata, orient='index')
    dfOBV.columns = ['OBV']
    dfOBV.index = pd.to_datetime(dfOBV.index)

    stockData['date'] = pd.to_datetime(stockData['date'])
    stockData.set_index('date', inplace=True)
    
    dfSMA = dfSMA.shift(1)
    dfRSI = dfRSI.shift(1)
    dfOBV = dfOBV.shift(1)

    combined_df = stockData.merge(dfSMA, left_index=True, right_index=True, how='left')
    combined_df = combined_df.merge(dfRSI, left_index=True, right_index=True, how='left')
    combined_df = combined_df.merge(dfOBV, left_index=True, right_index=True, how='left')


    combined_df = combined_df.reset_index()

    return combined_df
    

def splitData(processedData):
    processedData = processedData.apply(pd.to_numeric, errors='coerce')
    processedData = processedData.fillna(0)
    
    features = ['SMA', 'RSI', 'OBV', 'prev_close', 'prev_weekChange']
    x = processedData[features].values
    y = processedData['weekChangeBinary'].values

    train_size = int(0.8 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
        
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
     
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, x_train.shape[1]


def trainModel(trainData, inputSize):
    input_size = inputSize
    hidden_size = 50
    num_layers = 2
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch in trainData:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step(running_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainData):.4f}')
            
    return model, criterion


def evaluateModel(model, testData, criterion):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in testData:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            predicted = (outputs > 0.5).float()  
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        test_loss /= len(testData)
        accuracy = 100 * correct / total
    output = "loss: {}, accuracy: {}".format(test_loss, accuracy)
    return output

    
if __name__ == "__main__":
    main()
    