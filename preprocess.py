import os
import numpy as np
import librosa
import pandas as pd

# Path to the dataset
DATA_PATH = "data"

# List of genres in the dataset
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to extract features from an audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# Main function to preprocess the dataset
def preprocess_data(data_path, genres):
    features = []
    
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        for file_name in os.listdir(genre_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(genre_path, file_name)
                data = extract_features(file_path)
                if data is not None:
                    features.append([data, genre])
    
    features_df = pd.DataFrame(features, columns=['feature', 'label'])
    return features_df

# Run the preprocessing
if __name__ == "__main__":
    features_df = preprocess_data(DATA_PATH, GENRES)
    features_df.to_pickle('processed_features.pkl')
    print('Preprocessing completed. Processed data saved to processed_features.pkl')
