import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the dataset
n_samples = 10000  # Number of data points
time_intervals = np.arange(0, n_samples)  # Simulated time in seconds

# Simulate time series data for each feature
tension = np.sin(0.1 * time_intervals) * 100 + np.random.normal(0, 20, n_samples)
speed = np.sin(0.05 * time_intervals) * 10 + np.random.normal(0, 2, n_samples)
vibration = np.sin(0.2 * time_intervals) * 5 + np.random.normal(0, 1, n_samples)
temperature = np.sin(0.02 * time_intervals) * 50 + np.random.normal(0, 5, n_samples)
misalignment = np.sin(0.3 * time_intervals) * 0.5 + np.random.normal(0, 0.2, n_samples)

# Generate dislodgement labels (1 means dislodgement is imminent)
# Assume dislodgement happens when certain thresholds are crossed
labels = []
for i in range(n_samples):
    if (tension[i] > 120 or tension[i] < 80) and (vibration[i] > 6 or misalignment[i] > 0.7):
        labels.append(1)  # Imminent dislodgement
    else:
        labels.append(0)

# Create a DataFrame
data = pd.DataFrame({
    'Time': time_intervals,
    'Tension': tension,
    'Speed': speed,
    'Vibration': vibration,
    'Temperature': temperature,
    'Misalignment': misalignment,
    'Dislodged': labels  # Target label for dislodgement prediction
})

# Save to CSV for later use
data.to_csv('cable_belt_time_series_data.csv', index=False)

print("Synthetic time series data generated and saved as 'cable_belt_time_series_data.csv'.")
