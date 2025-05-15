import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- 1. Load the dataset ---
try:
    # Replace 'real_estate.csv' with the actual path to your dataset file
    df = pd.read_excel('estate_valuation.xlsx')
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: 'real_estate.csv' not found. Please update the file path.")
    # Exit if the file is not found
    exit()

# --- 2. Preprocess the data ---

# Identify features (X) and target (y) based on the provided columns
# Assuming the last column is the target 'Y house price of unit area'
# and the others (X1 to X6) are features.
# Let's use the simplified names from the image snippet for clarity
feature_names = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
                 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
target_name = 'Y house price of unit area'

X = df[feature_names]
y = df[target_name]

# Split data into training and testing sets
# Test set size is approximately 15-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Define the neural network model building function ---

def build_model(num_hidden_layers, neurons_per_layer, input_shape):
    """
    Builds a feed-forward neural network model.

    Args:
        num_hidden_layers (int): Number of hidden layers.
        neurons_per_layer (int): Number of neurons in each hidden layer.
        input_shape (int): Number of input features.

    Returns:
        tensorflow.keras.models.Sequential: The built model.
    """
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_shape=(input_shape,)))
    for _ in range(num_hidden_layers - 1): # Add additional hidden layers if num_hidden_layers > 1
        model.add(Dense(neurons_per_layer, activation='relu'))
    # Output layer for regression (1 neuron, no activation)
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- 4. & 5. Evaluate performance with different parameters and store results ---

experiments = []

# Define different hyperparameter combinations (at least 5)
param_combinations = [
    {'num_hidden_layers': 1, 'neurons_per_layer': 32, 'epochs': 50},
    {'num_hidden_layers': 1, 'neurons_per_layer': 64, 'epochs': 50},
    {'num_hidden_layers': 2, 'neurons_per_layer': 32, 'epochs': 50},
    {'num_hidden_layers': 2, 'neurons_per_layer': 64, 'epochs': 50},
    {'num_hidden_layers': 3, 'neurons_per_layer': 64, 'epochs': 100}, # More epochs for a deeper network
    {'num_hidden_layers': 1, 'neurons_per_layer': 32, 'epochs': 100}, # More epochs for a simple network
]

print("\nRunning experiments with different hyperparameters...")

for i, params in enumerate(param_combinations):
    print(f"\nExperiment {i+1}/{len(param_combinations)} with params: {params}")

    # Build the model
    model = build_model(params['num_hidden_layers'], params['neurons_per_layer'], X_train_scaled.shape[1])

    # Train the model (using a validation split to monitor overfitting)
    history = model.fit(X_train_scaled, y_train,
                        epochs=params['epochs'],
                        batch_size=32, # You can adjust batch size
                        validation_split=0.2, # Use a validation split of the training data
                        verbose=0) # Set verbose to 1 to see training progress

    # Evaluate the model on the test set
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Store results
    experiments.append({
        'Experiment No.': i + 1,
        'Number of Layers': params['num_hidden_layers'],
        'Number of Neurons (per layer)': params['neurons_per_layer'],
        'Number of Epochs': params['epochs'],
        'Test MSE': loss,
        'Test MAE': mae,
        'Training History': history.history # Store history for plotting
    })

    print(f"  Test MSE: {loss:.4f}, Test MAE: {mae:.4f}")

# --- 7. Formulate the comparison table ---

results_df = pd.DataFrame(experiments)
# Remove the 'Training History' column for the printed table
comparison_table = results_df[['Experiment No.', 'Number of Layers', 'Number of Neurons (per layer)',
                               'Number of Epochs', 'Test MSE', 'Test MAE']]

print("\n--- Comparison Table of Model Performance ---")
print(comparison_table.to_string(index=False))

# --- 8. Plot training history for selected experiments ---

print("\n--- Plotting Training History ---")

# You can choose which experiments to plot based on the table above
experiments_to_plot = [0, 3, 4] # Example: Plotting experiments 1, 4, and 5

for exp_index in experiments_to_plot:
    if exp_index < len(experiments):
        exp = experiments[exp_index]
        history = exp['Training History']
        epochs = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(12, 5))

        # Plot MSE (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['loss'], 'bo', label='Training MSE')
        plt.plot(epochs, history['val_loss'], 'b', label='Validation MSE')
        plt.title(f'Experiment {exp["Experiment No."]} - Training and Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        # Plot MAE (Accuracy equivalent for regression)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['mae'], 'bo', label='Training MAE')
        plt.plot(epochs, history['val_mae'], 'b', label='Validation MAE')
        plt.title(f'Experiment {exp["Experiment No."]} - Training and Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print(f"Warning: Experiment index {exp_index} out of range for plotting.")


# --- 9. Identify overfitting ---
print("\n--- Identifying Overfitting ---")
print("Analyze the plots above. Overfitting occurs when the validation loss (or MAE) starts to increase while the training loss (or MAE) continues to decrease.")
print("The point of overfitting is roughly where the validation curve starts to turn upwards.")


# --- 10. Train the final model on the full training data with validation split ---
# Choose the best performing model architecture based on the comparison table
# For demonstration, let's pick one that performed reasonably well (e.g., Experiment 4)
best_params = param_combinations[3] # Index 3 corresponds to Experiment 4

print(f"\n--- Training final model with best parameters ({best_params}) ---")

final_model = build_model(best_params['num_hidden_layers'], best_params['neurons_per_layer'], X_train_scaled.shape[1])

# Train on the full training data, using a validation split for monitoring
final_history = final_model.fit(X_train_scaled, y_train,
                                epochs=best_params['epochs'],
                                batch_size=32,
                                validation_split=0.2, # Still use validation split to monitor
                                verbose=1) # Set verbose to 1 to see training progress

# --- 11. Test the trained NS on the test set ---
print("\n--- Evaluating final model on the test set ---")
test_loss, test_mae = final_model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Final Model Test MSE: {test_loss:.4f}")
print(f"Final Model Test MAE: {test_mae:.4f}")

# --- 12. Output prediction results and desired responses for the test set ---
print("\n--- Predictions vs. Actual Values on Test Set ---")

# Get predictions
y_pred = final_model.predict(X_test_scaled).flatten() # Flatten to 1D array

# Display a few predictions vs actual
print("\nSample Predictions vs Actual:")
for i in range(10): # Display first 10 samples
    print(f"Sample {i+1}: Predicted = {y_pred[i]:.2f}, Actual = {y_test.iloc[i]:.2f}")

# --- 13. Display predictions and desired responses on a graph ---
print("\n--- Plotting Predictions vs. Actual Values ---")

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line for perfect prediction
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('Actual vs. Predicted House Prices on Test Set')
plt.grid(True)
plt.show()

# --- Conclusions ---
print("\n--- Conclusions ---")
print("Based on the comparison table, different hyperparameters resulted in varying performance.")
print(f"The final model trained with {best_params} achieved a Test MSE of {test_loss:.4f} and a Test MAE of {test_mae:.4f}.")
print("The plots of training history show how the model learned over epochs and can help identify potential overfitting.")
print("The scatter plot of actual vs. predicted values visually represents the model's accuracy on the test set.")
print("Ideally, the points on the scatter plot should lie close to the diagonal line.")
print("Further improvements could involve trying different network architectures, activation functions, optimizers, regularization techniques, or more extensive hyperparameter tuning.")