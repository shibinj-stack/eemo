# ============================================================
#  KeyMood LSTM — app.py
#  Single deployment: serves frontend + Flask API
#  Run:  python app.py
# ============================================================

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import tensorflow as tf

from feature_extractor import (
    extract_features_from_raw,
    features_to_vector,
    build_lstm_sequence,
    FEATURE_NAMES,
    EMOTION_LABELS,
    TIMESTEPS,
)

# Serve frontend static files from current directory
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ── Load model and scaler on startup ─────────────────────────
MODEL_PATH  = os.path.join('model', 'lstm_model.keras')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

print("Loading LSTM model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: model/lstm_model.keras not found.")
    exit(1)

if not os.path.exists(SCALER_PATH):
    print("ERROR: model/scaler.pkl not found.")
    exit(1)

model  = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print("Model loaded successfully")


# ═══════════════════════════════════════════════════════════════
# FRONTEND ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


# ═══════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body received'}), 400

    try:
        features_dict      = extract_features_from_raw(data)
        feature_vec        = features_to_vector(features_dict)
        feature_vec_scaled = scaler.transform(
            feature_vec.reshape(1, -1)
        ).flatten()
        sequence           = build_lstm_sequence(feature_vec_scaled)
        X                  = sequence[np.newaxis, :, :]
        probabilities      = model.predict(X, verbose=0)[0]
        predicted_idx      = int(np.argmax(probabilities))
        dominant_emotion   = EMOTION_LABELS[predicted_idx]
        confidence         = round(float(probabilities[predicted_idx]) * 100, 1)

        scores = {
            label: round(float(prob) * 100, 1)
            for label, prob in zip(EMOTION_LABELS, probabilities)
        }

        return jsonify({
            'emotion':    dominant_emotion,
            'confidence': confidence,
            'scores':     scores,
            'features':   {k: round(float(v), 3) for k, v in features_dict.items()},
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':   'ok',
        'model':    'LSTM Bidirectional',
        'emotions': EMOTION_LABELS,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
