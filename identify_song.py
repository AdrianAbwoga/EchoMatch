import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder

def load_new_sample_features(file_path):
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    return np.array(features).reshape(1, -1, 1)

def identify_song(new_sample_path, model_path='audio_cnn_model.h5', features_path='processed_features.pkl'):
    model = load_model(model_path)
    new_sample_features = load_new_sample_features(new_sample_path)

    data = pd.read_pickle(features_path)
    encoder = LabelEncoder()
    encoder.fit(data['label'].tolist())

    predictions = model.predict(new_sample_features)
    predicted_label = np.argmax(predictions, axis=1)
    song_label = encoder.inverse_transform(predicted_label)

    print(f"Identified song: {song_label[0]}")

if __name__ == "__main__":
    new_sample_features_path = 'new_sample_features.pkl'
    identify_song(new_sample_features_path)
