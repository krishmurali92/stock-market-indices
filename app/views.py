from flask import render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os
import pickle
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from app import app
import seaborn as sns

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    ticker = None
    models_performance = {}
    available_models = {}
    # Update available_models after training and saving a new model
    available_models = load_models()

    if request.method == 'POST':
        ticker = request.form.get('ticker')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        algo = request.form.get('algo')  # getting the selected algorithm
        test_size = request.form.get('test_size')
        filename = ticker + '.csv'
        
        data = yf.download(ticker, start=start_date, end=end_date)
        data.dropna(inplace=True)
        data.to_csv(filename)

        
        # Creating the plots
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            plt.figure(figsize=(14, 8))
            plt.plot(data.index, data[column])
            plt.title(f'{ticker} {column} Over Time')
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.grid(True)
            plt.savefig(f'app/static/{ticker}_{column}.png')

        # Creating a heatmap (correlation matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.savefig(f'app/static/{ticker}_heatmap.png')

        # Creating a histogram for 'Close' prices
        plt.figure(figsize=(10, 8))
        sns.histplot(data['Close'], kde=True)
        plt.title(f'{ticker} Close Price Distribution')
        plt.xlabel('Close Price')
        plt.ylabel('Frequency')
        plt.savefig(f'app/static/{ticker}_hist.png')
       
        # Assuming dataset has 'Open', 'High', 'Low', 'Close' and 'Volume' columns
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = data['Close'].shift(-1)  # Using next day's closing price as target
        y = y[:-1]  # Removing the last NaN value
        X = X[:-1]  # Making sure X and y have the same length

        algo = request.form.get('algo')
        model_name = request.form.get('model_name')

        test_size_str = request.form.get('test_size').split('/')[0]
        test_size = float(test_size_str)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if algo == 'linear':
            model = LinearRegression()
        elif algo == 'decision_tree':
            model = DecisionTreeRegressor()
        elif algo == 'random_forest':
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        models_performance[algo] = {'mse': mse, 'mae': mae, 'r2': r2}

        # Save model with pickle
        model_name = request.form.get('model_name')  # Get model name from form
        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Send the models_performance to your template...
    return render_template('index.html', ticker=ticker, models_performance=models_performance, models=available_models)

def load_models(directory='models'):
    model_files = os.listdir(directory)
    models = {}
    for file in model_files:
        model_name = os.path.splitext(file)[0]  # remove the .pkl extension
        with open(f'{directory}/{file}', 'rb') as f:
            models[model_name] = pickle.load(f)
    return models


@app.route('/predict', methods=['POST'])
def predict():
    open_price = float(request.form.get('open'))
    high = float(request.form.get('high'))
    low = float(request.form.get('low'))
    close = float(request.form.get('close'))
    volume = float(request.form.get('volume'))
    model_name = request.form.get('model_name')

    # No need for the previous model line, load directly from pickle file
    with open(f'models/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict([[open_price, high, low, close, volume]])
    models = load_models()

    return render_template('index.html', prediction=prediction, models=models)

