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
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
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
        features = np.array(features).reshape(1, -1)
    
        if features.shape[1] != 128:
            # Example padding to 128
            features = np.pad(features, ((0, 0), (0, 128 - features.shape[1])), 'constant')
        

        with open('new_sample_features.pkl', 'wb') as f:
            pickle.dump(features, f)
        prediction = model.predict(features)
        print(f"Model prediction: {prediction}")  # Debug statement
        label = encoder.inverse_transform([np.argmax(prediction)])
        print(f"Predicted label: {label[0]}")
    else:
        print("Failed to extract features.")

if __name__ == "__main__":
    # Relative path to the audio file
    audio_file_path = os.path.join('uploads', 'blues.00000.wav')
    process_and_save_sample(audio_file_path)