import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from stock_forecast import (fetch_historical_data, preprocess_data, build_model, 
                            train_model, plot_test_predictions, predict_next_month_daily,
                            plot_historical_and_predictions, DEFAULT_SEQ_LENGTH, FEATURES)

# Constants (default values)
CONFIG = {
    "ticker": "TSLA",  # Changed from AAPL to JPM to match the example in stock_forecast.py
    "start_date": "2020-01-01",
    "end_date": "2025-03-11",
    "epochs": 40,
    "learning_rate": 0.001,
    "units": 200,
    "dropout_rate": 0.2,
    "seq_length": 60,  # Using the default from stock_forecast
    "batch_size": 16,
    "num_tests": 4  # Increased from 1 to 3 for more robust testing
}

def run_single_test(params: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray, object, object]:
    """Run a single test with specified hyperparameters."""
    # Fetch and preprocess data
    stock_data = fetch_historical_data(params["ticker"], params["start_date"], params["end_date"])
    X_train, X_test, y_train, y_test, scaler, processed_data, features_list = preprocess_data(stock_data, params["seq_length"])

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, units=params["units"], dropout_rate=params["dropout_rate"], 
                        learning_rate=params["learning_rate"])

    # Train model
    trained_model, history = train_model(model, X_train, y_train, epochs=params["epochs"], 
                                         batch_size=params["batch_size"])
   
    # Plot predicted vs actual prices if requested
    if params.get("plot_results", False):
        plt.figure(figsize=(12, 6))
        plot_test_predictions(trained_model, X_test, y_test, scaler, features_list)
        plt.title(f'Predicted vs Actual Prices ({params["test_param"]} = {params[params["test_param"]]}, Test)')
        plt.show()
    
    # Get the best validation loss
    best_val_loss = min(history.history['val_loss'])
    return best_val_loss, X_test, y_test, scaler, trained_model, processed_data, features_list

def run_hyperparameter_tests(param_name: str, param_values: List[Any], base_config: Dict[str, Any] = CONFIG) -> Dict[str, List[float]]:
    """Run tests for a specified hyperparameter with given values."""
    results = {str(val): [] for val in param_values}
    
    for param_value in param_values:
        # Create a copy of the base config and update the test parameter
        test_config = base_config.copy()
        test_config[param_name] = param_value
        test_config["test_param"] = param_name  # Store the parameter being tested for plotting
        
        print(f"\nTesting {param_name} = {param_value}")
        for test_num in range(test_config["num_tests"]):
            print(f"  Running test {test_num + 1}/{test_config['num_tests']}...")
            val_loss, X_test, y_test, scaler, trained_model, processed_data, features_list = run_single_test(test_config)
            results[str(param_value)].append(val_loss)
            print(f"    Validation Loss: {val_loss:.6f}")
    
    return results

def analyze_results(results: Dict[str, List[float]], param_name: str) -> Tuple[str, float]:
    """Analyze test results and determine the best parameter value."""
    avg_losses = {}
    for param_value, losses in results.items():
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        avg_losses[param_value] = avg_loss
        print(f"\n{param_name} {param_value}:")
        print(f"  Individual Losses: {[f'{loss:.6f}' for loss in losses]}")
        print(f"  Average Validation Loss: {avg_loss:.6f}")
        print(f"  Standard Deviation: {std_loss:.6f}")

    # Find the best parameter value (lowest average validation loss)
    best_param_value = min(avg_losses, key=avg_losses.get)
    best_avg_loss = avg_losses[best_param_value]
    
    print(f"\nBest {param_name}: {best_param_value} with Average Validation Loss: {best_avg_loss:.6f}")
    return best_param_value, best_avg_loss

def plot_results(results: Dict[str, List[float]], param_name: str):
    """Plot the validation losses for each parameter value as a boxplot."""
    param_values = list(results.keys())
    losses = [results[val] for val in param_values]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(losses, labels=param_values)
    plt.title(f'Validation Loss Distribution Across {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.show()

def test_best_model(best_param_name: str, best_param_value: Any, config: Dict[str, Any] = CONFIG):
    """Test the best model and generate predictions."""
    print(f"\nTesting best model with {best_param_name} = {best_param_value}")
    
    # Create a copy of the config with the best parameter
    best_config = config.copy()
    best_config[best_param_name] = best_param_value
    best_config["plot_results"] = True  # Enable plotting for the best model
    
    # Run a single test with the best parameter
    val_loss, X_test, y_test, scaler, trained_model, processed_data, features_list = run_single_test(best_config)
    
    # Generate future predictions
    future_dates, predictions_real = predict_next_month_daily(
        trained_model, processed_data, scaler, features_list, seq_length=best_config["seq_length"], days=60
    )
    
    # Plot historical data and predictions
    plot_historical_and_predictions(
        stock_data=fetch_historical_data(best_config["ticker"], best_config["start_date"], best_config["end_date"]),
        scaler=scaler, 
        future_dates=future_dates, 
        predictions_real=predictions_real, 
        X_test=X_test, 
        y_test=y_test, 
        seq_length=best_config["seq_length"],
        features_list=features_list
    )
    
    # Print prediction values
    print("\nFuture Predictions (Next 60 Days):")
    for i, (date, price) in enumerate(zip(future_dates, predictions_real)):
        if i % 5 == 0:  # Print every 5th day to reduce output
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    return trained_model, future_dates, predictions_real

if __name__ == "__main__":
    # Example 1: Test learning rates
    param_to_test = "units"
    values_to_test = [200, 250, 300 ]
    
    # Run tests
    test_results = run_hyperparameter_tests(param_to_test, values_to_test)
    
    # Analyze results
    best_param, best_loss = analyze_results(test_results, param_to_test)
    
    # Plot boxplot of results
    plot_results(test_results, param_to_test)
    
    # Test the best model and generate predictions
    best_model, future_dates, predictions = test_best_model(param_to_test, float(best_param))
    
    # Example 2: Test different model architectures (units)
    # Uncomment to run this test
    """
    param_to_test = "units"
    values_to_test = [50, 100, 200]
    test_results = run_hyperparameter_tests(param_to_test, values_to_test)
    best_param, best_loss = analyze_results(test_results, param_to_test)
    plot_results(test_results, param_to_test)
    best_model, future_dates, predictions = test_best_model(param_to_test, int(best_param))
    """
    
    # Example 3: Test sequence lengths
    # Uncomment to run this test
    """
    param_to_test = "seq_length"
    values_to_test = [30, 60, 90]
    test_results = run_hyperparameter_tests(param_to_test, values_to_test)
    best_param, best_loss = analyze_results(test_results, param_to_test)
    plot_results(test_results, param_to_test)
    best_model, future_dates, predictions = test_best_model(param_to_test, int(best_param))
    """