# EchoMatch: Music Genre Recommendation System

EchoMatch is a music genre recommendation system that classifies songs into genres and provides tailored recommendations based on user-uploaded or recorded audio files. The system uses the GTZAN dataset and a deep learning model built with a Convolutional Neural Network (CNN).

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Validation and Evaluation](#validation-and-evaluation)
- [System Deployment](#system-deployment)
- [Future Enhancements](#future-enhancements)
- [How to Use](#how-to-use)
- [Acknowledgments](#acknowledgments)

---

## **Project Overview**
EchoMatch provides an intuitive web-based interface for music enthusiasts to:
1. Upload or record audio files.
2. Get the genre of the uploaded song.
3. Receive tailored song recommendations based on the predicted genre.

The project leverages audio feature extraction techniques, deep learning, and cloud deployment to create a seamless user experience.

---

## **Dataset**
**GTZAN Dataset**
- **Size**: 1,000 audio files (10 genres, 100 files per genre).
- **Genres**: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.
- **Audio Specifications**: WAV format, 30 seconds per file.
- **Why GTZAN?**: Balanced dataset with well-labeled genres, suitable for music classification research.

---

## **Preprocessing**
The preprocessing pipeline involves:
1. **Audio Loading**: Using Librosa to load audio files.
2. **Feature Extraction**: Extracting MFCCs (Mel Frequency Cepstral Coefficients) as 128-dimensional feature vectors.
3. **Label Encoding**: Converting genre labels into numerical format using `LabelEncoder`.
4. **Saving Features**: Storing extracted features and labels in `processed_features.pkl` for training and inference.

---

## **Model Training**
### **Model Architecture**
A CNN model was designed for genre classification:
- **Input**: 128 MFCC features.
- **Layers**:
  - Conv1D layers to extract patterns from audio data.
  - MaxPooling layers to reduce dimensionality.
  - Dense layers for classification.
- **Output**: Softmax activation for multi-class classification.

### **Training Details**
- **Optimizer**: Adam.
- **Loss Function**: Sparse categorical cross-entropy.
- **Metrics**: Accuracy.
- **Epochs**: 100.
- **Batch Size**: 32.

---

## **Validation and Evaluation**
### **Validation**
- **Split**: 80% training, 20% testing.
- **Techniques**: Stratified sampling to maintain genre distribution.

### **Evaluation Metrics**
- **Accuracy**: Achieved 69% accuracy on the test set.
- **F1-Score**: Evaluated per genre to measure performance balance.
- **Artifacts**:
  - Confusion matrix.
  - Precision-recall scores for each genre.

---



### **Workflow**
1. Users upload or record an audio file.
2. The system predicts the genre.
3. Recommendations are provided based on similar genres.



---

## **Future Enhancements**
- Expand the dataset to include more genres and larger audio samples.
- Enhance the recommendation system by integrating user feedback.
- Implement more sophisticated models (e.g., recurrent neural networks).

---

## **How to Use**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AdrianAbwoga/EchoMatch.git
   cd EchoMatch
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Web Interface**:
   - Open a browser and go to `http://localhost:5000`.

---

## **Acknowledgments**
- The GTZAN dataset creators for their work in building a benchmark for music classification.
- Open-source libraries such as TensorFlow, Librosa, and Flask for enabling rapid development.
- [Strathmore University] for providing resources and support.

---

Feel free to fork, contribute, and raise issues for enhancements! Thank you for exploring EchoMatch!

