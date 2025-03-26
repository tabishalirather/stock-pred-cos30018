import json
import numpy as np

def save_metrics(model, X_train, y_train, x_test, y_test, model_name, file_path='model_errors.json'):
    # Evaluate the model on training data
    train_loss, train_mse = model.evaluate(X_train, y_train, verbose=0)
    train_rmse = np.sqrt(train_mse)
    print(f"Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}")

    # Evaluate the model on test data
    test_loss, test_mse = model.evaluate(x_test, y_test, verbose=0)
    test_rmse = np.sqrt(test_mse)
    print(f"Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}")

    # Define the errors
    errors = {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "test_mse": test_mse,
        "test_rmse": test_rmse
    }

    # Create a dictionary to store the model name and errors
    error_data = {
        "model_name": model_name,
        "errors": errors
    }

    # Read existing data from the file if it exists
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