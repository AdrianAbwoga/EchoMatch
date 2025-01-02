import pickle
import os

# Set the absolute path to the file
file_path = os.path.abspath("../all_song_labels.pkl")  # Replace with the actual path

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(data)
except FileNotFoundError:
    print(f"The file {file_path} was not found.")
except EOFError:
    print("The file is empty or corrupted.")
