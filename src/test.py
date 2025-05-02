import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    test_data, model_path="./dnn_model.h5", input_shape=None, batch_size=32
):
    """
    Evaluate a TensorFlow regression model on test data.
    
    Args:
        model_path (str): Path to saved TensorFlow model (.h5 or saved_model directory)
        test_data: Test dataset. Can be one of:
                   - Tuple of (x_test, y_test) numpy arrays
                   - tf.data.Dataset object
                   - Directory path for image dataset (if regression on images)
        input_shape (tuple): Required if test_data is a directory path
        batch_size (int): Batch size for evaluation
    """
    # Load the model
    print(f"Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        raise ValueError("Could not load model. Check the model path.")
    
    model.summary()
    
    # Prepare test data
    if isinstance(test_data, tuple):
        # Test data provided as (x_test, y_test) numpy arrays
        x_test, y_test = test_data
        print(f"\nTest data shape: {x_test.shape}, {y_test.shape}")
        
    elif isinstance(test_data, tf.data.Dataset):
        # Test data already in tf.data.Dataset format
        test_dataset = test_data
        
    elif isinstance(test_data, str):
        # Test data is a directory path (for image datasets)
        print(f"\nLoading test images from {test_data}")
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_data,
            image_size=input_shape[:2],
            batch_size=batch_size,
            shuffle=False,
            label_mode='float'  # Important for regression
        )
    else:
        raise ValueError("Unsupported test data format")
    
    # Evaluate the model
    print("\nEvaluating model...")
    
    if isinstance(test_data, tuple):
        # For numpy arrays
        loss = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        y_pred = model.predict(x_test, batch_size=batch_size).flatten()
        y_true = y_test.flatten()
        
    else:
        # For tf.data.Dataset
        loss = model.evaluate(test_dataset, verbose=1)
        
        # Collect predictions and true values
        y_true = []
        y_pred = []
        for x, y in test_dataset:
            y_true.extend(y.numpy().flatten())
            y_pred.extend(model.predict(x, verbose=0).flatten())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("\nRegression Metrics:")
    print(f"Loss (Model Metric): {loss}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()
    
    # Plot error distribution
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Prediction Error Distribution')
    plt.show()
    
    return {
        'loss': loss,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'y_true': y_true,
        'y_pred': y_pred,
        'errors': errors
    }


# Example usage:
if __name__ == "__main__":
    # Example 1: With numpy arrays
    # Generate some synthetic data for demonstration
    # np.random.seed(42)
    # x_test = np.random.rand(1000, 10)  # 1000 samples, 10 features
    # true_weights = np.random.rand(10)
    # y_test = np.dot(x_test, true_weights) + np.random.normal(0, 0.1, 1000)
    # results = evaluate_regression_model('my_regression_model.h5', (x_test, y_test))
    
    # Example 2: With image directory (for regression on images)
    # results = evaluate_regression_model(
    #     model_path='my_image_regression_model.h5',
    #     test_data='path/to/test_images',
    #     input_shape=(224, 224, 3),
    #     batch_size=32
    # )
    
    print("Please modify the example usage with your actual paths and data")

# Example usage:
if __name__ == "__main__":
    
    # Example 1: With numpy arrays
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
    test = pd.read_csv('test.csv')
    testy = test.pop('Valor Venda R$')
    test = test
    test = test.drop(columns='Unnamed: 0')
    print(test.values)
    print(testy.values)
    results = evaluate_model((test.values, testy.values))

    # Example 2: With image directory
    # results = evaluate_model(
    #     model_path='my_model.h5',
    #     test_data='path/to/test_images',
    #     input_shape=(224, 224, 3),
    #     batch_size=32
    # )

    print("Please modify the example usage with your actual paths and data")
