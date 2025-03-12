import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
from sklearn import preprocessing as p 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import multiprocessing
import os



# --------------------------------------------------
# Step 1: Import Necessary Libraries
# --------------------------------------------------

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# Step 2: Data Collection
# --------------------------------------------------
def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    return data

# --------------------------------------------------
# Step 3: Data Preprocessing & Feature Engineering
# --------------------------------------------------
def preprocess_data(data):
    #1. Handle Missing Values
    data.ffill(inplace=True)

    #2. Engineer Features
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    
    #3. Normalize/Scale Feature
    features = ['Close', 'MA50', 'MA200']
    min_max_scaler = p.MinMaxScaler()    
    data[features] = min_max_scaler.fit_transform(data[features])
    #print(data[features])

    #4. create input sequences and labels for time series forcasting
    def create_sequences(data, seq_length):
    #data: Close, MA50, MA200
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)
    seq_length = 60
    feature_data = data[['Close', 'MA50', 'MA200']].dropna().values #drop NaN from MA's
    X, y = create_sequences(feature_data, seq_length)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    #print(X[61][0][0])
    #print(y[1])

    #5. Split data into training and testing sets
    #We will use a fixed about 80% train, 20%test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, min_max_scaler, data


# --------------------------------------------------
# Step 4: Model Definition
# --------------------------------------------------
def build_model(input_shape):
    model = Sequential()

    #units define layer capacity, ensures the layer outputs a sequence, 
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    #compile the model
    #Adaptive Moment Estimation
    #Sets the loss function to Mean Squared Error (MSE), measure average squared difference between predicted and actual prices
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model
# --------------------------------------------------
# Step 5: Model Training
# --------------------------------------------------
def train_model(model, X_train, y_train, epochs=70, batch_size=32):
    """
    Pseudocode:
    1. Train the model on the training data.
    2. Monitor performance via training history (loss, etc.).
    3. Return the trained model and training history.
    """
    # Define early stopping callback to monitor validation loss
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2,
        callbacks=[early_stop]
    )
    return model, history
    
# --------------------------------------------------
# Step 6: Making Predictions for Various Horizons
# --------------------------------------------------
def prepare_input_for_horizon(model, data, horizon_days, seq_length=60, features=['Close', 'MA50', 'MA200']):
    #get the last seq_length rows of scaled data

    recent_data = data[features].dropna().tail(seq_length).values #shape: (60,3)
    current_sequence = recent_data.copy()

    if horizon_days == 1: 
        #for 1 day ahead, just use the latest 60 days
        input_sequence = recent_data.reshape(1, seq_length, len(features))
    else:
        #for longer horizons, predict step-by-step
        
        for day in range (horizon_days):
            ''' 
            Debug code 

            input_seq = current_sequence[-seq_length:].reshape(1, seq_length, len(features))
            next_pred = model.predict(input_seq, verbose=0)[0, 0]
            real_price = scaler.inverse_transform([[next_pred, 0, 0]])[0, 0]
            print(f"Day {day + 1}: ${real_price:.2f}")'
            '''

            input_seq = current_sequence[-seq_length:].reshape(1, seq_length, len(features))
            next_pred = model.predict(input_seq)[0, 0]

            #Get the last 50 and 200 days of close prices
            close_history = np.concatenate([current_sequence[:, 0], [next_pred]])
            ma50 = np.mean(close_history[-50:])
            ma200 = np.mean(close_history[-200:]) if len(close_history) >= 200 else np.mean(close_history)
            
            new_row = np.array([next_pred, ma50, ma200])

            current_sequence = np.vstack((current_sequence[1:], new_row))
        input_sequence = current_sequence[-seq_length:].reshape(1, seq_length, len(features))
        
        
            

    return input_sequence

def predict_future_prices(model, data, horizons):
    """
    Pseudocode:
    1. For each forecast horizon (tomorrow, week, month, etc.):
    2. Prepare the input sequence based on the most recent data.
    3. Use the model to predict the future stock price.
    4. Store and return predictions in a dictionary.
    """
    predictions = {}
    """
    Prepare input for horizons function
    """
    

    for horizon_name, horizon_days in horizons.items():
        input_sequence = prepare_input_for_horizon(model, data, horizon_days)  # Custom function
        pred = model.predict(input_sequence)[0, 0]
        predictions[horizon_name] = pred

    
    return predictions


# --------------------------------------------------
# Step 7: Investment Evaluation
# --------------------------------------------------
def evaluate_investment(predictions, current_price):
    """
    Pseudocode:
    1. Compare predicted prices with the current price.
    2. Calculate expected returns over various horizons.
    3. Optionally factor in risk metrics (volatility, drawdowns, etc.).
    4. Return a recommendation (e.g., "Good long term investment" or "Not recommended").
    """
    # Example pseudocode:
    # expected_return_1yr = (predictions['1_year'] - current_price) / current_price
    # if expected_return_1yr > some_threshold and risk_metrics are acceptable:
    #     recommendation = "Good long term investment"
    # else:
    #     recommendation = "Not recommended"
    # return recommendation
    pass

# --------------------------------------------------
# Graph train and test Data
# --------------------------------------------------

def plot_historical_close_real(data, scaler):
    scaled_close = data['Close'].values
    dummy = np.zeros_like(scaled_close)

    combined_scaled = np.column_stack((scaled_close, dummy, dummy))
    real_close = scaler.inverse_transform(combined_scaled)[:, 0]

    plt.figure(figsize=(12,6))
    plt.plot(data.index, real_close, label='Real Close Price (Inverse Transformed)')
    plt.title('AAPL Historical Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_train_test_data(data, train_size, seq_length):
    """
    Plots the entire Close price series with training and testing sections in different colors.
    
    Parameters:
    - data (DataFrame): Processed stock data with original Close prices.
    - train_size (int): Number of sequences in the training set.
    - seq_length (int): Length of each sequence.
    """

    #original close prices
    close_prices = data['Close'].dropna()
    total_train_points = train_size + seq_length

    #split train and test data from original close prices
    train_data = close_prices[:total_train_points]
    test_data = close_prices[total_train_points:]
    #print(f"train_data_shape: {train_data.shape}, test_data shape: {test_data.shape}")
    
    
    #plot 
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    plt.plot(test_data.index, test_data, label='Testing Data', color='orange')
    plt.title('Stock Close Price for AAPL: Training vs Testing Split')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------------------
# Recent 60 Days plot
# --------------------------------------------------
def plot_recent_60_days(stock_data, scaler):
    recent_scaled_close = stock_data['Close'].dropna().tail(60).values
    recent_scaled_close = recent_scaled_close.flatten()

    dummy_array = np.zeros((recent_scaled_close.shape[0], 3))
    dummy_array[:, 0] = recent_scaled_close
    recent_real_close = scaler.inverse_transform(dummy_array)[:, 0]


    plt.figure(figsize=(10, 5))
    plt.plot(processed_data.index[-60:], recent_real_close, marker='o', label='Real Close Price')
    plt.title("AAPL Last 60 Days Stock Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    
    plt.xticks(rotation=45)
   
    plt.legend()
    
    plt.tight_layout()
    # Display the plot
    plt.show()


# --------------------------------------------------
# Plot Test Predictions vs Actual
# --------------------------------------------------
def plot_test_predictions(model, X_test, y_test, scaler):
    # 1. Predict on the test set using the model.
    predictions_scaled = model.predict(X_test)
    
    # 2. Inverse transform the scaled predictions and actual test prices.
    # Note: The scaler was fit on 3 features ['Close', 'MA50', 'MA200'], so we only transform the first element (Close).
    predictions_real = []
    actual_real = []
    for pred_scaled, actual_scaled in zip(predictions_scaled, y_test):
        # Inverse transform prediction: Create a dummy vector where only the first value (Close) is used.
        pred_price = scaler.inverse_transform([[pred_scaled[0], 0, 0]])[0, 0]
        actual_price = scaler.inverse_transform([[actual_scaled, 0, 0]])[0, 0]
        predictions_real.append(pred_price)
        actual_real.append(actual_price)
    
    # 3. Plot the actual and predicted prices for comparison.
    plt.figure(figsize=(12, 6))
    plt.plot(actual_real, label='Actual Prices')
    plt.plot(predictions_real, label='Predicted Prices')
    plt.title('Test Set: Actual Prices vs Predicted Prices')
    plt.xlabel('Test Sample')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------------------
# first 100 predictions
# --------------------------------------------------

def debug_first_100_predictions(model, X_test, y_test, scaler, test_indices=None):
    # 1. Predict the scaled prices on the test set.
    predictions_scaled = model.predict(X_test)
    
    # 2. Inverse transform the predictions and actual values.
    #    Note: The scaler was applied on three features ['Close', 'MA50', 'MA200'],
    #    so we create a dummy vector to only recover the 'Close' price.
    predictions_real = []
    actual_real = []
    for pred_scaled, actual_scaled in zip(predictions_scaled, y_test):
        # Inverse transform returns an array; we take the first element (the 'Close' price).
        pred_price = scaler.inverse_transform([[pred_scaled[0], 0, 0]])[0, 0]
        actual_price = scaler.inverse_transform([[actual_scaled, 0, 0]])[0, 0]
        predictions_real.append(pred_price)
        actual_real.append(actual_price)
    
    # 3. Print the first 100 values with indices.
    print("Index            | Predicted Price | Actual Price")
    print("----------------------------------------------------")
    num_print = min(300, len(predictions_real))
    for i in range(num_print):
        # Use the provided test_indices if available; otherwise, use a sequential counter.
        idx = test_indices[i] if test_indices is not None else i + 1
        print(f"{idx!s:15} | {predictions_real[i]:14.2f} | {actual_real[i]:12.2f}")


if __name__ == "__main__":

    
    stock_data = fetch_historical_data("NVDA", "2020-01-01", "2025-03-11")
    '''
    #plot last 60 close data
    recent_data = stock_data['Close'].tail(60)
    plt.plot(recent_data, label='Last 60 Days of AAPL Close Price')
    plt.title('Recent 60-Day Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    '''

    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(stock_data)

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))  # (60, 3)
    trained_model, history = train_model(model, X_train, y_train)
    
    
    

    horizons = {"tomorrow": 1, "week":7, "month": 30}
    predictions = predict_future_prices(trained_model, processed_data, horizons)
    print("Scaled Predictions:", predictions)

    for horizon, pred in predictions.items():
        real_price = scaler.inverse_transform([[pred, 0, 0]])[0, 0]  # Only Close matters
        print(f"{horizon}: ${real_price:.2f}")

    #debug first 100

    
    debug_first_100_predictions(model, X_test, y_test, scaler)

    #plots test vs predictions
    plot_test_predictions(trained_model, X_test, y_test, scaler)

    #prints teh current price of stock
    current_price = stock_data['Close'].iloc[-1].item()
    real_price = scaler.inverse_transform([[current_price, 0, 0]])[0, 0]
    print(f"Current Price: ${real_price:.2f}")


    #plots predicted vs real data
    
    
    
    
    #plot last 60 days
    plot_recent_60_days(stock_data, scaler)

    #plot histocial data
    plot_historical_close_real(stock_data, scaler)

    #plot train and test
    train_size = len(X_train)
    plot_train_test_data(stock_data, train_size, seq_length=60)

    
    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    plot_test_predictions(trained_model, X_test, y_test, scaler)
    
    
    

    



