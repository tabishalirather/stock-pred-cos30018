import json
import numpy as np

def save_metrics_kans(model, X_train, y_train, X_test, y_test, model_name, file_path='model_errors.json'):
    # Create dictionaries for evaluation (the evaluate() method expects keys 'test_input' and 'test_label')
    train_data = {"test_input": X_train, "test_label": y_train}
    test_data = {"test_input": X_test, "test_label": y_test}

    # Evaluate the model on training and test data
    train_metrics = model.evaluate(train_data)
    test_metrics = model.evaluate(test_data)

    # Print the raw evaluation dictionaries
    print(f"train_metrics: {train_metrics}")
    print(f"test_metrics: {test_metrics}")

    # Extract RMSE from the evaluation dictionaries
    train_rmse = train_metrics['test_loss']
    test_rmse = test_metrics['test_loss']
    # Compute MSE by squaring the RMSE values
    train_mse = train_rmse ** 2
    test_mse = test_rmse ** 2

    # Print the computed error metrics
    print(f"Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}")
    print(f"Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}")

    # Prepare the error dictionary for saving
    errors = {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "test_mse": test_mse,
        "test_rmse": test_rmse
    }

    error_data = {
        "model_name": model_name,
        "errors": errors
    }

    # Load existing data if the file exists
    try:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
            if not isinstance(existing_data, list):
                existing_data = []
    except FileNotFoundError:
        existing_data = []

    # Append the new error data
    existing_data.append(error_data)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    print(f"Errors written to {file_path}")
