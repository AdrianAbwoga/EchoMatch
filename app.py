from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from process_new_sample import process_and_save_sample

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PLOT_FOLDER'] = 'static/plots/'

# Ensure the plot folder exists
if not os.path.exists(app.config['PLOT_FOLDER']):
    os.makedirs(app.config['PLOT_FOLDER'])

# Load preprocessed song features and labels separately
try:
    with open('all_song_features.pkl', 'rb') as f:
        all_song_features = pickle.load(f)

    with open('all_song_labels.pkl', 'rb') as f:
        all_song_labels = pickle.load(f)
except (FileNotFoundError, EOFError) as e:
    print(f"Error loading pickle files: {e}")
    raise RuntimeError(
        "Ensure `preprocess.py` has been run successfully, "
        "and that `all_song_features.pkl` and `all_song_labels.pkl` exist and are valid."
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files['audio']
    
    # Always save the uploaded file as 'audio.wav'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
    audio_file.save(file_path)
    
    # Process the uploaded audio file and save features
    predicted_label = process_and_save_sample(file_path)
    
    # Get recommendations based on the uploaded song
    recommendations = get_recommendations(file_path, predicted_label)

    # Visualize prediction and overwrite existing plot
    plot_file = plot_similarity(file_path)

    # Return the song label, recommendations, and the path to the plot image
    return jsonify({'song': predicted_label, 'recommendations': recommendations, 'plot': plot_file})

def get_recommendations(uploaded_file_path, predicted_label, top_n=5):
    # Load features of the uploaded song
    if not os.path.exists('new_sample_features.pkl'):
        raise FileNotFoundError(
            "new_sample_features.pkl not found. Ensure `process_and_save_sample` has run successfully."
        )
    with open('new_sample_features.pkl', 'rb') as f:
        uploaded_song_features = pickle.load(f).reshape(1, -1)

    # Filter songs by the predicted genre
    genre_specific_songs = {filename: features for filename, features in all_song_features.items() if all_song_labels[filename] == predicted_label}

    # If no songs match the predicted genre, fall back to the entire dataset
    if not genre_specific_songs:
        print(f"No songs found for predicted genre: {predicted_label}. Using the entire dataset for recommendations.")
        genre_specific_songs = all_song_features

    # Compute similarity within the filtered dataset
    similarities = [
        (filename, cosine_similarity(uploaded_song_features, np.array(features).reshape(1, -1))[0, 0])
        for filename, features in genre_specific_songs.items()
    ]

    # Sort songs by similarity score in descending order
    sorted_songs = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Prepare the top `n` recommendations
    recommendations = [f"{filename} ({all_song_labels[filename]})" for filename, _ in sorted_songs[:top_n]]
    return recommendations

def plot_similarity(uploaded_file_path):
    # Load features of the uploaded song
    if not os.path.exists('new_sample_features.pkl'):
        raise FileNotFoundError("new_sample_features.pkl not found.")
    with open('new_sample_features.pkl', 'rb') as f:
        uploaded_song_features = pickle.load(f).reshape(1, -1)

    # Calculate similarities with all genres
    genre_similarities = {}
    for genre in set(all_song_labels.values()):  # Check similarity with each genre
        genre_features = [
            features for filename, features in all_song_features.items() if all_song_labels[filename] == genre
        ]
        genre_avg_feature = np.mean(genre_features, axis=0)  # Average feature for the genre
        similarity = cosine_similarity(uploaded_song_features, genre_avg_feature.reshape(1, -1))[0, 0]
        genre_similarities[genre] = similarity

    # Create a radar chart for genre similarity
    genres = list(genre_similarities.keys())
    similarities = list(genre_similarities.values())

    # Sort the genres by similarity for better visualization
    sorted_genres = sorted(zip(genres, similarities), key=lambda x: x[1], reverse=True)
    sorted_genres = list(zip(*sorted_genres))

    # Number of genres
    num_vars = len(sorted_genres[0])
    
    # Compute angle for each genre on the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Create a radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, sorted_genres[1], color='skyblue', alpha=0.25)
    ax.plot(angles, sorted_genres[1], color='blue', linewidth=2)

    ax.set_yticklabels([])  # Hide radial ticks
    ax.set_xticks(angles)
    ax.set_xticklabels(sorted_genres[0], fontweight='bold', fontsize=12)

    ax.set_title('Genre Similarity to Uploaded Song', fontsize=14)
    plt.tight_layout()

    # Save the plot as an image (always overwrite the existing plot)
    plot_filename = 'static/plots/genre_similarity_radar_chart.png'
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(app.config['PLOT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
