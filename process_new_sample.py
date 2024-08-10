import os
import numpy as np
import librosa
import pickle
import tensorflow as tf  # Import TensorFlow

# Load pre-trained model and encoder
model = tf.keras.models.load_model('audio_cnn_model.h5')
encoder = pickle.load(open('processed_features.pkl', 'rb'))

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file
        # Example: Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)  # Mean of MFCC features over time
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_and_save_sample(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        label = encoder.inverse_transform([np.argmax(prediction)])
        print(f"Predicted label: {label[0]}")
    else:
        print("Failed to extract features.")

# Example usage
if __name__ == "__main__":
    file_path = 'uploads/audio.wav'  # Path to the uploaded file
    process_and_save_sample(file_path)
