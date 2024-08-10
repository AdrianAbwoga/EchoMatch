from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from process_new_sample import process_and_save_sample
from identify_song import identify_song

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files['audio']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
    audio_file.save(file_path)
    
    # Process the uploaded audio file
    process_and_save_sample(file_path)
    
    # Identify the song
    song_label = identify_song('new_sample_features.pkl')

    # For demonstration, we'll return the identified song label as a recommendation
    recommendations = get_recommendations(song_label)

    return jsonify({'song': song_label, 'recommendations': recommendations})

def get_recommendations(song_label):
    # Dummy recommendations for demonstration
    recommendations = ["Song A", "Song B", "Song C"]
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
