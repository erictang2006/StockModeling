import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
from sklearn import preprocessing as p 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


#Constants
DEFAULT_SEQ_LENGTH = 90
FEATURES = ['Close', 'MA50', 'MA200', 'Volatility_14', 'ATR_14', 'RSI', 'MACD', 'Signal', 'MACD_Hist']

# --------------------------------------------------
# Step 1: Import Necessary Libraries
# --------------------------------------------------

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# Step 2: Data Collection
# --------------------------------------------------
def fetch_historical_data(ticker: str, start_date: str, end_date: str):

    return yf.download(ticker, start=start_date, end=end_date)

# --------------------------------------------------
# Step 3: Data Preprocessing & Feature Engineering
# --------------------------------------------------
def preprocess_data(data, seq_length: int = DEFAULT_SEQ_LENGTH):
    #1. Handle Missing Values
    data.ffill(inplace=True)

    #2. Engineer Features
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # --Volatility Measure--

    # Calculate daiy returns (precentage change from the previous day)
    data['Returns'] = data ['Close'].pct_change()

    # Historical Volatility (14-DAY standard deviation of returns)
    data['Volatility_14'] = data['Returns'].rolling(window=14).std() * np.sqrt(252)

    # Average True Range (ATR). True range = The maximum of high-low, high-previous close, and low-previous close
    data['High_Low'] = data['High'] - data['Low']
    data['High_Close'] = np.abs(data['High'] - data['Close'].shift())
    data['Low_Close'] = np.abs(data['Low'] - data['Close'].shift())
    data['TR'] = np.maximum(data['High_Low'], np.maximum(data['High_Close'], data['Low_Close']))
    data['ATR_14'] = data['TR'].rolling(window=14).mean()
    

    # --Momentum Indicators--

    ## Relative Strength Index (RSI)
    # Calculate difference in price from previous step
    delta = data['Close'].diff()
    # Gains (if any) and losses (if any)
    gain = delta.clip(lower = 0)
    loss = -delta.clip(upper = 0)

    # Calculate rolling average of gains and losses with a 14-day window
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    #relative strength
    rs = avg_gain / avg_loss

    #rsi formula
    data['RSI'] = 100 - (100 / (1 + rs))

    ## MOving Average Convergence Divergence (MACD)
    # Compute the 12-day and 26-day EMAs (Exponential Movign Average) of the closing price
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    # MACD line
    data['MACD'] = ema12 - ema26
    # Signal line: 9-day EMA of the MACD line
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    # MACD Histogram: Difference between MACD line and Signal line
    data['MACD_Hist'] = data['MACD'] - data['Signal']


    # Update FEATURES list to include new volatility measures
    FEATURES = ['Close', 'MA50', 'MA200', 'Volatility_14', 'ATR_14', 'RSI', 'MACD', 'Signal', 'MACD_Hist']

    data = data.dropna()

    #3. Normalize/Scale Feature
    scaler = p.MinMaxScaler()
    data[FEATURES] = scaler.fit_transform(data[FEATURES])
    #print(data[features])

    #4. create input sequences and labels for time series forcasting
    def create_sequences(data, seq_length: int):
    #data: Close, MA50, MA200
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)
    
    feature_data = data[FEATURES].dropna().values #drop NaN from MA's
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

    return X_train, X_test, y_train, y_test, scaler, data, FEATURES


# --------------------------------------------------
# Step 4: Model Definition
# --------------------------------------------------
def build_model(input_shape, units: int = 200, dropout_rate: float = 0.2, learning_rate: float = 0.001):
    model = Sequential()

    #units define layer capacity, ensures the layer outputs a sequence, 
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 2, activation='relu'))  # Add this intermediate Dense layer
    model.add(Dense(1))

    #compile the model
    #Adaptive Moment Estimation
    #Sets the loss function to Mean Squared Error (MSE), measure average squared difference between predicted and actual prices
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='mean_squared_error')
    return model
# --------------------------------------------------
# Step 5: Model Training
# --------------------------------------------------
def train_model(model, X_train, y_train, epochs: int = 60, batch_size: int = 16):
    """
    Pseudocode:
    1. Train the model on the training data.
    2. Monitor performance via training history (loss, etc.).
    3. Return the trained model and training history.
    """
    # Define early stopping callback to monitor validation loss
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=14, 
        restore_best_weights=True)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history
    
# --------------------------------------------------
# Step 6: Making Predictions for Various Horizons
# --------------------------------------------------
def prepare_input_for_horizon(model, data, horizon_days, features_list=None, seq_length: int = DEFAULT_SEQ_LENGTH):
    #get the last seq_length rows of scaled data

    if features_list is None:
        features_list = FEATURES

    recent_data = data[features_list].dropna().tail(seq_length).values #shape: (60,3)
    current_sequence = recent_data.copy()

    if horizon_days == 1: 
        #for 1 day ahead, just use the latest 60 days
        input_sequence = recent_data.reshape(1, seq_length, len(features_list))
    else:
        #for longer horizons, predict step-by-step
        
        for _ in range (horizon_days):
            ''' 
            Debug code 

            input_seq = current_sequence[-seq_length:].reshape(1, seq_length, len(features))
            next_pred = model.predict(input_seq, verbose=0)[0, 0]
            real_price = scaler.inverse_transform([[next_pred, 0, 0]])[0, 0]
            print(f"Day {day + 1}: ${real_price:.2f}")'
            '''

            input_seq = current_sequence[-seq_length:].reshape(1, seq_length, len(features_list))
            next_pred = model.predict(input_seq)[0, 0]

            last_row = current_sequence[-1].copy()

            # Update Close price (first feature)
            new_close = next_pred

            # Estimate the new row based on predicted close price
            new_row = last_row.copy()
            new_row[0] = new_close  # Update Close

            # Update MA features if they exist in the feature list
            if len(features_list) > 2:  # If we have more than just Close
                # Update MA50 (simple approximation)
                if 'MA50' in features_list:
                    ma50_idx = features_list.index('MA50')
                    # Simple approximation: slightly adjust previous MA with new close
                    new_row[ma50_idx] = 0.98 * last_row[ma50_idx] + 0.02 * new_close
                
                # Update MA200 (simple approximation)
                if 'MA200' in features_list:
                    ma200_idx = features_list.index('MA200')
                    # Even smaller adjustment for slower MA
                    new_row[ma200_idx] = 0.995 * last_row[ma200_idx] + 0.005 * new_close
                
                # Update volatility measures if they exist
                if 'Volatility_14' in features_list:
                    vol_idx = features_list.index('Volatility_14')
                    # Keep volatility the same - this is a simplification
                    new_row[vol_idx] = last_row[vol_idx]
                
                if 'ATR_14' in features_list:
                    atr_idx = features_list.index('ATR_14')
                    # Keep ATR the same - this is a simplification  
                    new_row[atr_idx] = last_row[atr_idx]
            
            # Add the new row to the sequence
            current_sequence = np.vstack((current_sequence[1:], new_row))
            
        # Use the final sequence for prediction
        input_sequence = current_sequence[-seq_length:].reshape(1, seq_length, len(features_list))
    
    return input_sequence
        
        
            

    return input_sequence

def predict_future_prices(model, data, horizons, features_list=None):
    """
    Pseudocode:
    1. For each forecast horizon (tomorrow, week, month, etc.):
    2. Prepare the input sequence based on the most recent data.
    3. Use the model to predict the future stock price.
    4. Store and return predictions in a dictionary.
    """
    if features_list is None:
        features_list = FEATURES
        
    predictions = {}
    
    for horizon_name, horizon_days in horizons.items():
        input_sequence = prepare_input_for_horizon(model, data, horizon_days, features_list)
        pred = model.predict(input_sequence, verbose=0)[0, 0]
        predictions[horizon_name] = pred
    
    return predictions

# Add this new function below your existing predict_future_prices
def predict_next_month_daily(model, data, scaler, features_list=None, seq_length: int = DEFAULT_SEQ_LENGTH, days: int = 90):
    """
    Predicts daily stock prices for the next 30 days, step-by-step.
    Returns dates and predicted prices (in real dollars).
    """

    if features_list is None:
        features_list = FEATURES

    # Get the last 60 days of scaled data
    recent_data = data[features_list].dropna().tail(seq_length).values  # Shape: (60, 3)
    current_sequence = recent_data.copy()
    
    # Generate future dates starting from the last date in the data
    last_date = data.index[-1]  # Should be March 11, 2025
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
    
    # Store predictions
    predictions_scaled = []
    
    # Predict day-by-day for 30 days
    for _ in range(days):
        input_seq = current_sequence[-seq_length:].reshape(1, seq_length, len(features_list))
        next_pred = model.predict(input_seq, verbose=0)[0, 0]  # Predict next day's scaled Close
        predictions_scaled.append(next_pred)
        
        # Create a new row with the predicted close and approximated other features
        last_row = current_sequence[-1].copy()
        new_row = last_row.copy()
        new_row[0] = next_pred  # Update Close price
        # Update other features with simple approximations (same logic as prepare_input_for_horizon)

        if 'MA50' in features_list:
            ma50_idx = features_list.index('MA50')
            new_row[ma50_idx] = 0.98 * last_row[ma50_idx] + 0.02 * next_pred
            
        if 'MA200' in features_list:
            ma200_idx = features_list.index('MA200')
            new_row[ma200_idx] = 0.995 * last_row[ma200_idx] + 0.005 * next_pred
        
        # Keep volatility measures the same for simplicity
        
        # Add the new row to the sequence
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    # Convert scaled predictions back to real prices
    predictions_real = []
    for pred in predictions_scaled:
         # Create dummy array with zeros for all features except Close
        dummy = np.zeros(len(features_list))
        dummy[0] = pred  # Set Close value
        real_price = scaler.inverse_transform([dummy])[0, 0]
        predictions_real.append(real_price)
    
    return future_dates, predictions_real



# --------------------------------------------------
# Step 7: Investment Evaluation
# --------------------------------------------------
def evaluate_investment(predictions, current_price, scaler):
    """
    Pseudocode:
    1. Compare predicted prices with the current price.
    2. Calculate expected returns over various horizons.
    3. Optionally factor in risk metrics (volatility, drawdowns, etc.).
    4. Return a recommendation (e.g., "Good long term investment" or "Not recommended").
    """
    # Inverse transform the current scaled price to its original value
    current_price_real = scaler.inverse_transform([[current_price, 0, 0]])[0,0]

    #Inverse transform the scaled predictions for 1 month and 1 year horizons
    month_pred_real = scaler.inverse_transform([[predictions['month'], 0, 0]])[0, 0]
    year_pred_real = scaler.inverse_transform([[predictions['1_year'], 0, 0]])[0, 0]

    # Define thresholds (example: require at least a 2% gain for 1 month and 15% for 1 year)
    

    # Calculate expected returns as percentages
    expected_return_month = (month_pred_real - current_price_real) / current_price_real
    expected_return_year = (year_pred_real - current_price_real) / current_price_real
    
    month_threshold = 0.02
    year_threshold = 0.15

    # Determine recommendation based on expected returns
    if expected_return_year > year_threshold and expected_return_month > month_threshold:
        recommendation = "Good long term investment"
    else:
        recommendation = "Not recommended"

    # Return a summary dictionary with the evaluation results
    return {
        "current_price": current_price_real,
        "month_prediction": month_pred_real,
        "year_prediction": year_pred_real,
        "expected_return_month": expected_return_month,
        "expected_return_year": expected_return_year,
        "recommendation": recommendation
    }

# --------------------------------------------------
# Graph train and test Data
# --------------------------------------------------

def plot_historical_close_real(data, scaler, ticker, features_list=None):
    """
    Plots the historical close prices in their original (non-scaled) values.
    
    Parameters:
    - data: DataFrame containing the scaled stock data (from preprocess_data)
    - scaler: The MinMaxScaler used to scale the data
    - features_list: List of features used in scaling (defaults to global FEATURES)
    """
    if features_list is None:
        features_list = FEATURES
    
    # Get the scaled Close prices (assumes data is the scaled dataframe)
    scaled_close = data['Close'].values.astype(float)
    
    # Extract scaling parameters for the Close feature (assumed to be the first column)
    close_min = scaler.data_min_[0]
    close_range = scaler.data_range_[0]
    
    # Reconstruct original Close prices manually
    real_close = scaled_close * close_range + close_min
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, real_close, label='Historical Close Price')
    plt.title(ticker + " Historical Close Price")
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_train_test_data(data, train_size, seq_length, ticker):
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
    plt.title('Stock Close Price for ' + ticker +  'Training vs Testing Split')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------------------
# Recent 60 Days plot
# --------------------------------------------------
def plot_recent_60_days(stock_data, scaler, ticker, features_list=None):
    """
    Plots the most recent 60 days of close prices in their original (non-scaled) values.
    
    Parameters:
    - stock_data: DataFrame containing the scaled stock data
    - scaler: The MinMaxScaler used to scale the data
    - features_list: List of features used in scaling (defaults to global FEATURES)
    """
    if features_list is None:
        features_list = FEATURES
    
    # Get the recent scaled Close prices and ensure it's flattened to a 1D array
    recent_scaled_close = stock_data['Close'].dropna().tail(60).values.flatten()
    
    # Create dummy arrays for inverse transformation with the right number of features
    dummy_array = np.zeros((len(recent_scaled_close), len(features_list)))
    
    # Set the Close price (first column) with our scaled values
    dummy_array[:, 0] = recent_scaled_close
    
    # Inverse transform to get original prices
    recent_real_close = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Get the appropriate date indices for the recent data
    recent_dates = stock_data.index[-60:]
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(recent_dates, recent_real_close, marker='o', label='Real Close Price')
    plt.title(ticker +  "Last 60 Days Stock Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Plot Test Predictions vs Actual
# --------------------------------------------------
def plot_test_predictions(model, X_test, y_test, scaler, features_list=None):
    # 1. Predict on the test set using the model.
    if features_list is None:
        features_list = FEATURES

    predictions_scaled = model.predict(X_test)
    
    # 2. Inverse transform the scaled predictions and actual test prices.
    # Note: The scaler was fit on 3 features ['Close', 'MA50', 'MA200'], so we only transform the first element (Close).
    predictions_real = []
    actual_real = []
    for pred_scaled, actual_scaled in zip(predictions_scaled, y_test):
        # For predictions: Create dummy array with zeros for all features except Close
        dummy_pred = np.zeros(len(features_list))
        dummy_pred[0] = pred_scaled[0]
        pred_price = scaler.inverse_transform([dummy_pred])[0, 0]

        # For actual values: Create dummy array with zeros for all features except Close
        dummy_actual = np.zeros(len(features_list))
        dummy_actual[0] = actual_scaled
        actual_price = scaler.inverse_transform([dummy_actual])[0, 0]
        
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

def debug_first_100_predictions(model, X_test, y_test, scaler, features_list=None, test_indices=None):
    # 1. Predict the scaled prices on the test set.
    predictions_scaled = model.predict(X_test)
    
    # If features_list is not provided, use the global FEATURES list
    if features_list is None:
        features_list = FEATURES
    
    # 2. Inverse transform the predictions and actual values.
    predictions_real = []
    actual_real = []
    for pred_scaled, actual_scaled in zip(predictions_scaled, y_test):
        # For predictions: Create dummy array with zeros for all features except Close
        dummy_pred = np.zeros(len(features_list))
        dummy_pred[0] = pred_scaled[0]  # Set Close value
        pred_price = scaler.inverse_transform([dummy_pred])[0, 0]
        
        # For actual values: Create dummy array with zeros for all features except Close
        dummy_actual = np.zeros(len(features_list))
        dummy_actual[0] = actual_scaled  # Set Close value
        actual_price = scaler.inverse_transform([dummy_actual])[0, 0]
        
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


# --------------------------------------------------
# plot next 60 days
# --------------------------------------------------

def plot_historical_and_predictions(stock_data, scaler, future_dates, predictions_real, X_test, y_test, ticker, seq_length: int = DEFAULT_SEQ_LENGTH, days_history: int = 60, features_list=None):
    """
    Plots:
    - Last days_history days of historical closing prices.
    - Test set predictions vs actual prices.
    - Predicted prices for the next 30 days.
    All as jagged lines.
    """
    if features_list is None:
        features_list = FEATURES
    # 2. Test set predictions vs actual
    test_predictions_scaled = model.predict(X_test, verbose=0)
    # Convert test predictions to real prices
    test_predictions_real = []
    test_actual_real = []
    
    for pred_scaled, actual_scaled in zip(test_predictions_scaled, y_test):
        # For predictions
        dummy_pred = np.zeros(len(features_list))
        dummy_pred[0] = pred_scaled[0]
        pred_price = scaler.inverse_transform([dummy_pred])[0, 0]
        
        # For actual values
        dummy_actual = np.zeros(len(features_list))
        dummy_actual[0] = actual_scaled
        actual_price = scaler.inverse_transform([dummy_actual])[0, 0]
        
        test_predictions_real.append(pred_price)
        test_actual_real.append(actual_price)
    
    # Get the corresponding dates for the test set
    train_size = len(stock_data) - len(X_test) - seq_length
    test_start_idx = train_size + seq_length
    test_dates = stock_data.index[test_start_idx:test_start_idx + len(X_test)]
    
    # Plot everything
    plt.figure(figsize=(14, 7))
    # Test actual
    plt.plot(test_dates, test_actual_real, '-', label='Test Actual Prices', color='blue')
    # Test predictions
    plt.plot(test_dates, test_predictions_real, '--', label='Test Predicted Prices', color='orange')
    # Future predictions
    plt.plot(future_dates, predictions_real, '--', label=f'Future Predicted Prices (Next {len(predictions_real)} Days)', color='red')
    
    plt.title('Stock Price: Historical, Test Predictions vs Actual, and Future Prediction for ' + ticker)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# plots  MACD, signal, and historgram 
# --------------------------------------------------
def plot_macd(data, ticker):
    """
    Plots the MACD line, Signal line, and MACD Histogram for the last 5 months.
    Histogram bars are colored green for positive values and red for negative values.
    
    Parameters:
    - data: DataFrame containing at least 'MACD', 'Signal', and 'MACD_Hist' columns with a datetime index.
    """
    # Filter the data to only include the last 5 months
    six_months_ago = data.index[-1] - pd.DateOffset(months=6)
    data_6m = data.loc[data.index >= six_months_ago]
    
    # Create a figure with 2 subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot MACD and Signal lines on the first subplot
    ax1.plot(data_6m.index, data_6m['MACD'], label='MACD', color='blue', linewidth=1.5)
    ax1.plot(data_6m.index, data_6m['Signal'], label='Signal', color='orange', linewidth=1.5)
    ax1.set_title("MACD and Signal Lines (Last 6 Months) for " + ticker)
    ax1.set_ylabel("MACD Value")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Close price on the second subplot
    ax2.plot(data_6m.index, data_6m['Close'], label='Close Price', color='green', linewidth=1.5)
    ax2.set_title("Stock Close Price (Last 6 Months) for " + ticker)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    ticker = 'SOFI'
    
    horizons = {
        "tomorrow": 1, 
        "week":7, 
        "month": 30,
        "3_months": 90
        #"6_months": 182,
        #"1_year": 365
        }
    
    # Fetch historical stock data from 2020-01-01 to 2025-03-11
    stock_data = fetch_historical_data(ticker, "2020-01-01", "2025-03-16")
    
    # Preprocess the data and engineer features (e.g., moving averages, scaling)
    X_train, X_test, y_train, y_test, scaler, processed_data, features_list = preprocess_data(stock_data)

    # Build the LSTM model with the specified input shape
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))  # (60, 3)

    # Train the model on the training data
    trained_model, history = train_model(model, X_train, y_train)
    
    # Make predictions for each specified horizon using the most recent processed data
    predictions = predict_future_prices(trained_model, processed_data, horizons, features_list)

     # Extract the current scaled "Close" price from the processed data
    current_price = stock_data['Close'].iloc[-1].item()

    # Evaluate the investment using the predictions for 1 month and 1 year
    
    # Display results
    print("Scaled Predictions:", predictions)
    for horizon, pred in predictions.items():
        # Create dummy array for inverse transform
        dummy = np.zeros(len(features_list))
        dummy[0] = pred  # Set Close value
        real_price = scaler.inverse_transform([dummy])[0, 0]
        print(f"{horizon}: ${real_price:.2f}")

    #debug first 100
    #debug_first_100_predictions(model, X_test, y_test, scaler)

    #plots test vs predictions
    plot_test_predictions(trained_model, X_test, y_test, scaler)

    # Predict daily values for the next month
    # Plot historical data (last 60 days) and predictions
    future_dates, predictions_real = predict_next_month_daily(trained_model, processed_data, scaler, features_list)
    plot_historical_and_predictions(stock_data, scaler, future_dates, predictions_real, X_test, y_test, ticker)

    # current_price from raw data is already unscaled
    current_price = stock_data['Close'].iloc[-1].item()
    print(f"Current Price: ${current_price:.2f}")

    # Predict daily prices for the next 90 days
    future_dates, predictions_real = predict_next_month_daily(trained_model, processed_data, scaler, features_list, days=90)

    # Print the future predicted prices
    print("Future predicted prices for the next 90 days:")
    for date, price in zip(future_dates, predictions_real):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")


    
    # Print out the investment evaluation results, including expected returns and recommendation
    '''
    evaluation = evaluate_investment(predictions, current_price, scaler)
    print("Current Price: $", float(evaluation["current_price"]))
    print("1 Month Prediction: $", float(evaluation["month_prediction"]))
    print("1 Year Prediction: $", float(evaluation["year_prediction"]))
    print("Expected Return (Month):", float(evaluation["expected_return_month"]))
    print("Expected Return (Year):", float(evaluation["expected_return_year"]))
    print("Recommendation:", evaluation["recommendation"])
    '''

    #plot last 60 days
    #plot_recent_60_days(stock_data, scaler, ticker)

    #plot histocial data
    plot_historical_close_real(stock_data, scaler, ticker)

    #plot train and test
    #train_size = len(X_train)
    #plot_train_test_data(stock_data, train_size, seq_length=60, ticker)

    #plots macd
    plot_macd(processed_data, ticker)
    
    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    plot_historical_and_predictions(stock_data, scaler, future_dates, predictions_real, X_test, y_test, ticker)