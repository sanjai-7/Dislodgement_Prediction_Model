import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the time series data
data = pd.read_csv('cable_belt_time_series_data.csv')

# Features and target
features = ['Tension', 'Speed', 'Vibration', 'Temperature', 'Misalignment']
X = data[features]
y = data['Dislodged']

# Normalize the data for better performance with LSTM
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM: [samples, time steps, features]
time_steps = 30  # Number of time steps to consider for each prediction
X_series = []
y_series = []
for i in range(len(X_scaled) - time_steps):
    X_series.append(X_scaled[i:i+time_steps])
    y_series.append(y[i + time_steps])

X_series = np.array(X_series)
y_series = np.array(y_series)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_series, y_series, test_size=0.2, random_state=42)


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Predict the next time steps and alert if dislodgement is predicted
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert predictions to binary 0 or 1

# Print an alert if dislodgement is predicted
if np.any(y_pred == 1):
    print("Alert: Imminent dislodgement detected!")
else:
    print("System is stable.")
