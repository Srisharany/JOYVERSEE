from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
from emotion_transformer import load_model_and_predict
from sklearn.preprocessing import LabelEncoder
from emotion_transformer import load_model_and_predict


# Flask app
app = Flask(__name__)

# Emotion label decoding
label_encoder = LabelEncoder()
label_encoder.fit(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    keypoints = np.array(data['keypoints']).reshape(1, -1).astype(np.float32)
    input_tensor = torch.tensor(keypoints, dtype=torch.float32)

    emotion_index = load_model_and_predict(input_tensor)
    emotion_label = label_encoder.inverse_transform([emotion_index])[0]

    return jsonify({'emotion': emotion_label})

if __name__ == '__main__':
    app.run(debug=True)
