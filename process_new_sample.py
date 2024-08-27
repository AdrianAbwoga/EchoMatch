import librosa
import numpy as np
import pickle
import tensorflow as tf
import os

# Load the trained model and encoder
model = tf.keras.models.load_model('audio_cnn_model.h5')
encoder = pickle.load(open('encoder.pkl', 'rb'))

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0)
        print(f"Extracted features: {features}")  # Debug statement
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_and_save_sample(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    features = extract_features(file_path)
    if features is not None:
        # Ensure features length matches the model's expected input length
        target_length = 128  # The length used during training
        if features.shape[0] < target_length:
            # Pad the features if they are shorter
            features = np.pad(features, (0, target_length - features.shape[0]), 'constant')
        elif features.shape[0] > target_length:
            # Truncate the features if they are longer
            features = features[:target_length]
        
        # Reshape the features for the model
        features = np.array(features).reshape(1, -1, 1)
        
        # Predict the label
        prediction = model.predict(features)
        print(f"Model prediction: {prediction}")  # Debug statement
        
        label = encoder.inverse_transform([np.argmax(prediction)])
        print(f"Predicted label: {label[0]}")
    else:
        print("Failed to extract features.")

if __name__ == "__main__":
    # Relative path to the audio file
    audio_file_path = os.path.join('uploads', 'classical.00000.wav')
    process_and_save_sample(audio_file_path)
