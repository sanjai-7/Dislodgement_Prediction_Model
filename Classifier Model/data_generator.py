import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the dataset
n_samples = 10000  # Number of data samples
n_features = 5  # Number of features: Tension, Speed, Vibration, Temperature, Misalignment

# Generate synthetic data
# Tension (high tension can indicate dislodgement risk)
tension = np.random.normal(100, 20, n_samples)  # mean=100, std=20

# Speed (higher or lower than usual can indicate a problem)
speed = np.random.normal(10, 2, n_samples)  # mean=10 m/s, std=2

# Vibration (higher vibration might indicate wear or misalignment)
vibration = np.random.normal(5, 1, n_samples)  # mean=5, std=1

# Temperature (abnormal high temperature can lead to issues)
temperature = np.random.normal(50, 5, n_samples)  # mean=50°C, std=5

# Misalignment (small values indicate slight misalignment)
misalignment = np.random.normal(0.5, 0.2, n_samples)  # mean=0.5 (aligned), std=0.2

# Generate labels (0: no dislodgement, 1: risk of dislodgement)
# We use a simple rule-based logic to generate dislodgement risk
labels = []
for i in range(n_samples):
    if (tension[i] > 120 or tension[i] < 80) and (vibration[i] > 6 or misalignment[i] > 0.7):
        labels.append(1)  # High tension or misalignment combined with high vibration
    else:
        labels.append(0)

# Create a DataFrame to store the synthetic data
data = pd.DataFrame({
    'Tension': tension,
    'Speed': speed,
    'Vibration': vibration,
    'Temperature': temperature,
    'Misalignment': misalignment,
    'Dislodged': labels  # Target label
})

# Save to CSV for later use
data.to_csv('cable_belt_synthetic_data.csv', index=False)

print("Synthetic data generated and saved as 'cable_belt_synthetic_data.csv'.")
