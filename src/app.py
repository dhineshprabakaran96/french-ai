import os
import io
import random
import base64
import hashlib
import secrets
import tempfile
import struct
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from Levenshtein import ratio
from gtts import gTTS
from pydub import AudioSegment

# Initialize Flask app, configure static folder for frontend
app = Flask(__name__, static_folder='frontend')
CORS(app)

# ===========================
# LOAD MODEL & SETUP
# ===========================

# Remove Windows specific library path setting
# Just install 'espeak-ng' in your container; phonemizer finds it automatically on Linux
# EspeakWrapper.set_library(...)  # DO NOT set on Linux

model_name = "facebook/wav2vec2-large-xlsr-53-french"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

# ===========================
# UTILITY FUNCTIONS
# ===========================

def get_words_from_csv(file_path):
    words = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        for line in csvfile:
            row = line.strip().split(',')
            words.extend(row)
    return [word.strip() for word in words if word.strip()]

def transcribe_audio(audio_data, sample_rate):
    if sample_rate != 16000:
        ratio = 16000 / sample_rate
        new_length = int(len(audio_data) * ratio)
        audio_data = np.interp(np.linspace(0, len(audio_data), new_length), np.arange(len(audio_data)), audio_data)
        sample_rate = 16000

    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    # Normalize audio if needed (clip to [-1,1])
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val

    input_values = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding="longest").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.lower()

def assess_pronunciation(reference_text, user_transcription):
    try:
        reference_phonemes = phonemize(reference_text, language='fr-fr', backend='espeak', strip=True, njobs=1)
        user_phonemes = phonemize(user_transcription, language='fr-fr', backend='espeak', strip=True, njobs=1)

        similarity_score = ratio(reference_phonemes, user_phonemes)
        accuracy = similarity_score * 100
        feedback = "Excellent pronunciation!" if similarity_score > 0.8 else "Your pronunciation needs improvement."

        return {
            "accuracy": round(accuracy, 2),
            "feedback": feedback,
            "reference_phonemes": reference_phonemes,
            "user_phonemes": user_phonemes,
            "transcription": user_transcription
        }
    except Exception as e:
        return {"error": str(e)}

# ===========================
# DATA STORAGE (In-Memory)
# ===========================

# Load words dataset file from project folder
french_words = get_words_from_csv("frenchWordsDataset.csv")

practice_history = []

users_db = {}       # {email: {username, email, password (hashed), created_at}}
user_sessions = {}  # {session_token: {email, username, login_time}}

# ===========================
# AUTH HELPERS
# ===========================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def generate_session_token():
    return secrets.token_hex(32)

# ===========================
# FRONTEND ROUTES (Serve Static Files)
# ===========================

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/homepage.html')
def serve_homepage():
    return send_from_directory(app.static_folder, 'homepage.html')

@app.route('/learnFrench.html')
def serve_learn():
    return send_from_directory(app.static_folder, 'learnFrench.html')

@app.route('/history.html')
def serve_history():
    return send_from_directory(app.static_folder, 'history.html')

@app.route('/signUP.html')
def serve_signup():
    return send_from_directory(app.static_folder, 'signUP.html')

@app.route('/<path:path>')
def serve_static(path):
    # Serve static assets for front-end (CSS, JS, images, etc)
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return jsonify({"error": "File not found"}), 404

# ===========================
# API ENDPOINTS
# ===========================

@app.route('/api/words/random', methods=['GET'])
def get_random_word():
    if not french_words:
        return jsonify({"error": "No words dataset available"}), 500
    word = random.choice(french_words)
    return jsonify({"word": word})

@app.route('/api/words/all', methods=['GET'])
def get_all_words():
    return jsonify({"words": french_words})

@app.route('/api/pronunciation/reference', methods=['POST'])
def generate_reference_audio():
    data = request.get_json()
    word = data.get('word')

    if not word:
        return jsonify({"error": "Word is required"}), 400

    try:
        tts = gTTS(text=word, lang='fr', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        return jsonify({"audio": audio_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pronunciation/assess', methods=['POST'])
def assess_user_pronunciation():
    temp_file_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400

        audio_file = request.files['audio']
        reference_word = request.form.get('word')

        if not reference_word:
            return jsonify({"error": "Reference word is required"}), 400

        # Save uploaded audio file to a temp file
        temp_fd, temp_file_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        audio_file.save(temp_file_path)

        # Convert audio to WAV 16k mono for processing
        converted_path = temp_file_path.replace('.wav', '_converted.wav')

        try:
            audio_segment = AudioSegment.from_file(temp_file_path)

            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_frame_rate(16000)
            audio_segment = audio_segment.set_sample_width(2)

            audio_segment.export(converted_path, format="wav")
            processing_path = converted_path

        except Exception:
            # Fallback if pydub conversion fails: use original file
            processing_path = temp_file_path

        speech, rate = sf.read(processing_path)

        user_transcription = transcribe_audio(speech, rate)

        assessment = assess_pronunciation(reference_word, user_transcription)

        if not assessment.get('error'):
            practice_entry = {
                "word": reference_word,
                "accuracy": assessment["accuracy"],
                "feedback": assessment["feedback"],
                "date": str(datetime.now().date()),
                "time": str(datetime.now().time().strftime("%H:%M"))
            }
            practice_history.append(practice_entry)

        return jsonify(assessment)

    except Exception as e:
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

    finally:
        # Clean up temp files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        converted_path = temp_file_path.replace('.wav', '_converted.wav') if temp_file_path else None
        if converted_path and os.path.exists(converted_path):
            try:
                os.unlink(converted_path)
            except Exception:
                pass


@app.route('/api/pronunciation/phonetic', methods=['POST'])
def get_word_phonetic():
    data = request.get_json()
    word = data.get('word')

    if not word:
        return jsonify({"error": "Word is required"}), 400

    try:
        phonetic = phonemize(word, language='fr-fr', backend='espeak', strip=True, njobs=1)
        return jsonify({"word": word, "phonetic": phonetic})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/practice/history', methods=['GET'])
def get_practice_history():
    return jsonify({"history": practice_history})

@app.route('/api/practice/stats', methods=['GET'])
def get_practice_stats():
    if not practice_history:
        return jsonify({
            "total_words": 0,
            "total_attempts": 0,
            "average_accuracy": 0,
            "current_streak": 0
        })

    total_attempts = len(practice_history)
    total_accuracy = sum(entry["accuracy"] for entry in practice_history)
    average_accuracy = round(total_accuracy / total_attempts, 1) if total_attempts > 0 else 0

    unique_words = len(set(entry["word"] for entry in practice_history))

    current_streak = 0
    for entry in reversed(practice_history):
        if entry["accuracy"] > 80:
            current_streak += 1
        else:
            break

    return jsonify({
        "total_words": unique_words,
        "total_attempts": total_attempts,
        "average_accuracy": average_accuracy,
        "current_streak": current_streak
    })

# ===========================
# AUTHENTICATION ENDPOINTS
# ===========================

@app.route('/api/auth/register', methods=['POST'])
def register_user():
    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not all([username, email, password]):
        return jsonify({"error": "Username, email, and password are required"}), 400

    if email in users_db:
        return jsonify({"error": "Email already registered"}), 400

    users_db[email] = {
        "username": username,
        "email": email,
        "password": hash_password(password),
        "created_at": str(datetime.now())
    }

    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/auth/login', methods=['POST'])
def login_user():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if email not in users_db:
        return jsonify({"error": "Invalid email or password"}), 401

    user = users_db[email]

    if not verify_password(password, user["password"]):
        return jsonify({"error": "Invalid email or password"}), 401

    session_token = generate_session_token()
    user_sessions[session_token] = {
        "email": email,
        "username": user["username"],
        "login_time": str(datetime.now())
    }

    return jsonify({
        "message": "Login successful",
        "session_token": session_token,
        "username": user["username"]
    }), 200

@app.route('/api/auth/user', methods=['GET'])
def get_user_info():
    session_token = request.headers.get('Authorization')

    if not session_token or session_token not in user_sessions:
        return jsonify({"error": "Invalid or expired session"}), 401

    session_data = user_sessions[session_token]
    email = session_data["email"]
    user = users_db[email]

    return jsonify({
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"]
    }), 200

@app.route('/api/auth/logout', methods=['POST'])
def logout_user():
    session_token = request.headers.get('Authorization')

    if session_token and session_token in user_sessions:
        del user_sessions[session_token]

    return jsonify({"message": "Logout successful"}), 200

# ===========================
# RUN APP
# ===========================

if __name__ == '__main__':
    # Use port from environment variable PORT or default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
