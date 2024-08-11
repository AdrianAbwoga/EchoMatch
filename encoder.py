import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your training dataset
data = pd.read_pickle('processed_features.pkl')  # Replace with your actual training data file

# Initialize and fit the LabelEncoder
encoder = LabelEncoder()
encoder.fit(data['label'])

# Save the encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
