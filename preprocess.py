import os
import numpy as np
import librosa
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Path to the dataset
DATA_PATH = "data"

# List of genres in the dataset
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to extract features from an audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# Main function to preprocess the dataset
def preprocess_data(data_path, genres):
    features = []
    all_song_features = {}
    all_song_labels = {}

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        for file_name in os.listdir(genre_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(genre_path, file_name)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    all_song_features[file_name] = data
                    all_song_labels[file_name] = genre

    return features, all_song_features, all_song_labels

if __name__ == "__main__":
    # Extract features and labels
    raw_features, all_song_features, all_song_labels = preprocess_data(DATA_PATH, GENRES)

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(raw_features)

    # Save normalized features for all songs
    with open('all_song_features.pkl', 'wb') as f:
        pickle.dump(all_song_features, f)
    print("Normalized song features saved to all_song_features.pkl")

    # Save labels for all songs
    with open('all_song_labels.pkl', 'wb') as f:
        pickle.dump(all_song_labels, f)
    print("Song labels saved to all_song_labels.pkl")

    # Save the scaler for runtime normalization
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")
