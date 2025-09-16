import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Generate Synthetic IoT Data (Simulating multiple IoT devices)
def generate_iot_data(num_devices=3, num_samples=1000):
    """Generate synthetic IoT data for multiple devices."""
    data = []
    for device_id in range(num_devices):
        # Simulate normal sensor data (e.g., temperature, humidity)
        temp = np.random.normal(25, 2, num_samples)  # Normal temperature ~25°C
        humidity = np.random.normal(50, 5, num_samples)  # Normal humidity ~50%
        
        # Add anomalies (e.g., 10% of data is anomalous)
        anomaly_indices = random.sample(range(num_samples), int(0.1 * num_samples))
        for idx in anomaly_indices:
            temp[idx] += random.choice([-10, 10])  # Anomalous temperature spikes
            humidity[idx] += random.choice([-20, 20])  # Anomalous humidity spikes
        
        device_data = pd.DataFrame({
            'device_id': [device_id] * num_samples,
            'temperature': temp,
            'humidity': humidity
        })
        data.append(device_data)
    return data

# 2. Build Autoencoder Model for Anomaly Detection
def build_autoencoder(input_dim):
    """Build an autoencoder model for anomaly detection."""
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# 3. Train Local Model on a Device
def train_local_model(autoencoder, data, epochs=10):
    """Train an autoencoder on a device's data."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['temperature', 'humidity']])
    
    autoencoder.fit(scaled_data, scaled_data,
                    epochs=epochs,
                    batch_size=32,
                    verbose=0)
    return autoencoder, scaler

# 4. Aggregate Models (Federated Averaging)
def aggregate_models(global_model, local_models):
    """Aggregate weights from local models using Federated Averaging."""
    global_weights = global_model.get_weights()
    local_weights = [model.get_weights() for model in local_models]
    
    # Average weights
    avg_weights = [np.mean([w[i] for w in local_weights], axis=0) for i in range(len(global_weights))]
    global_model.set_weights(avg_weights)
    return global_model

# 5. Detect Anomalies
def detect_anomalies(autoencoder, scaler, data, threshold=2.0):
    """Detect anomalies using reconstruction error."""
    scaled_data = scaler.transform(data[['temperature', 'humidity']])
    reconstructions = autoencoder.predict(scaled_data, verbose=0)
    mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
    
    # Mark anomalies where reconstruction error > threshold
    anomalies = mse > threshold
    return anomalies, mse

# 6. Federated Learning Simulation
def federated_learning_simulation(num_devices=3, num_rounds=5, epochs=10):
    """Simulate federated learning for IoT anomaly detection."""
    print("Starting Federated Learning for IoT Anomaly Detection")
    
    # Generate synthetic IoT data
    device_data = generate_iot_data(num_devices=num_devices)
    
    # Initialize global model
    input_dim = 2  # Temperature and humidity
    global_model = build_autoencoder(input_dim)
    
    # Federated Learning Rounds
    for round in range(num_rounds):
        print(f"\nRound {round + 1}/{num_rounds}")
        local_models = []
        local_scalers = []
        
        # Train local model on each device
        for device_id in range(num_devices):
            print(f"Training on Device {device_id}...")
            local_model = build_autoencoder(input_dim)
            local_model.set_weights(global_model.get_weights())  # Start with global weights
            data = device_data[device_id]
            trained_model, scaler = train_local_model(local_model, data, epochs=epochs)
            local_models.append(trained_model)
            local_scalers.append(scaler)
        
        # Aggregate local models
        print("Aggregating local models...")
        global_model = aggregate_models(global_model, local_models)
        
        # Evaluate on each device
        for device_id in range(num_devices):
            data = device_data[device_id]
            anomalies, mse = detect_anomalies(global_model, local_scalers[device_id], data)
            num_anomalies = np.sum(anomalies)
            print(f"Device {device_id}: Detected {num_anomalies} anomalies")
    
    # Save final global model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_model.save(f"iot_anomaly_global_model_{timestamp}.h5")
    print(f"Global model saved as iot_anomaly_global_model_{timestamp}.h5")
    
    return global_model, local_scalers, device_data

# 7. Main Function
def main():
    """Run the IoT anomaly detection with federated learning."""
    num_devices = 3
    num_rounds = 5
    epochs = 10
    
    print("IoT Device Anomaly Detection with Federated Learning")
    global_model, scalers, device_data = federated_learning_simulation(
        num_devices=num_devices, num_rounds=num_rounds, epochs=epochs
    )
    
    # Final evaluation and visualization
    print("\nFinal Evaluation:")
    for device_id in range(num_devices):
        data = device_data[device_id]
        anomalies, mse = detect_anomalies(global_model, scalers[device_id], data)
        print(f"\nDevice {device_id} Anomaly Report:")
        print(f"Total Anomalies Detected: {np.sum(anomalies)}")
        print(f"Average Reconstruction Error: {np.mean(mse):.4f}")
        
        # Display sample anomalies
        anomaly_data = data[anomalies]
        if not anomaly_data.empty:
            print("Sample Anomalous Data Points:")
            print(anomaly_data.head())

if __name__ == "__main__":
    main()
