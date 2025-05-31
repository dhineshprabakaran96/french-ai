from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from Levenshtein import ratio
import sounddevice as sd
import soundfile as sf
import random
import time
import subprocess
from gtts import gTTS
from pydub import AudioSegment
import io
import csv
import json
import sqlite3
from datetime import datetime, timedelta
import base64
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
CORS(app, supports_credentials=True)

# eSpeak configuration
EspeakWrapper.set_library('C:\\Program Files\\eSpeak NG\\libespeak-ng.dll')
os.environ['ESPEAK_DATA_PATH'] = "C:\\Program Files\\eSpeak NG"

# Load the French-specific model and processor
model_name = "facebook/wav2vec2-large-xlsr-53-french"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

# Database initialization
def init_db():
    conn = sqlite3.connect('french_phonetics.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Practice sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS practice_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            word TEXT NOT NULL,
            phonetic TEXT NOT NULL,
            user_transcription TEXT,
            accuracy REAL,
            feedback TEXT,
            session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Words table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE NOT NULL,
            phonetic TEXT NOT NULL,
            difficulty_level TEXT DEFAULT 'beginner'
        )
    ''')
    
    conn.commit()
    conn.close()

# Load words from CSV and populate database
def populate_words_db():
    conn = sqlite3.connect('french_phonetics.db')
    cursor = conn.cursor()
    
    # Sample French words with phonetics (you can expand this)
    sample_words = [
        ("bonjour", "/bɔ̃.ʒuʁ/", "beginner"),
        ("merci", "/mɛʁ.si/", "beginner"),
        ("au revoir", "/o.ʁə.vwaʁ/", "beginner"),
        ("comment", "/kɔ.mɑ̃/", "beginner"),
        ("parler", "/paʁ.le/", "intermediate"),
        ("français", "/fʁɑ̃.sɛ/", "intermediate"),
        ("baguette", "/ba.gɛt/", "intermediate"),
        ("fromage", "/fʁɔ.maʒ/", "intermediate"),
        ("pomme", "/pɔm/", "beginner"),
        ("fenêtre", "/fə.nɛtʁ/", "advanced"),
        ("voiture", "/vwa.tyʁ/", "intermediate"),
        ("maison", "/mɛ.zɔ̃/", "beginner"),
    ]
    
    for word, phonetic, difficulty in sample_words:
        cursor.execute('INSERT OR IGNORE INTO words (word, phonetic, difficulty_level) VALUES (?, ?, ?)', 
                      (word, phonetic, difficulty))
    
    conn.commit()
    conn.close()

# Helper functions
def transcribe_audio(audio_data, sample_rate=16000):
    try:
        # Ensure audio is a 1D array
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Process the audio
        input_values = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding="longest").input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription.lower()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def phonetic_comparison(reference_text, user_transcription):
    try:
        # Convert to phonemes using the espeak backend
        reference_phonemes = phonemize(reference_text, language='fr-fr', backend='espeak', strip=True, njobs=1)
        user_phonemes = phonemize(user_transcription, language='fr-fr', backend='espeak', strip=True, njobs=1)
        
        # Compare phonemes
        similarity_score = ratio(reference_phonemes, user_phonemes)
        accuracy = similarity_score * 100
        
        # Generate feedback
        if similarity_score > 0.85:
            feedback = "Excellent pronunciation!"
        elif similarity_score > 0.70:
            feedback = "Good pronunciation, minor improvements needed."
        elif similarity_score > 0.50:
            feedback = "Fair pronunciation, practice the vowel sounds."
        else:
            feedback = "Needs improvement. Focus on each syllable."
        
        return accuracy, feedback, reference_phonemes, user_phonemes
    
    except Exception as e:
        return 0, f"Error analyzing pronunciation: {str(e)}", "", ""

def generate_audio(text):
    try:
        tts = gTTS(text=text, lang='fr', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_data = base64.b64encode(fp.read()).decode('utf-8')
        return audio_data
    except Exception as e:
        print(f"Audio generation error: {e}")
        return None

# API Routes

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Validate email format (basic validation)
    if '@gmail.com' not in email:
        return jsonify({'error': 'Please use a Gmail address'}), 400
    
    # Validate password requirements
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    try:
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE email = ? OR username = ?', (email, username))
        if cursor.fetchone():
            return jsonify({'error': 'User already exists'}), 409
        
        # Create new user
        password_hash = generate_password_hash(password)
        cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)', 
                      (username, email, password_hash))
        conn.commit()
        
        return jsonify({'message': 'User created successfully'}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not all([email, password]):
        return jsonify({'error': 'Missing email or password'}), 400
    
    try:
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, username, password_hash FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return jsonify({
                'message': 'Login successful',
                'user': {
                    'id': user[0],
                    'username': user[1],
                    'email': email
                }
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/user/stats', methods=['GET'])
def get_user_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    
    try:
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        # Get total words practiced
        cursor.execute('SELECT COUNT(DISTINCT word) FROM practice_sessions WHERE user_id = ?', (user_id,))
        words_practiced = cursor.fetchone()[0]
        
        # Get current streak (days)
        cursor.execute('''
            SELECT DATE(session_date) as practice_date 
            FROM practice_sessions 
            WHERE user_id = ? 
            ORDER BY session_date DESC
        ''', (user_id,))
        dates = [row[0] for row in cursor.fetchall()]
        
        streak = 0
        if dates:
            current_date = datetime.now().date()
            for date_str in dates:
                practice_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if (current_date - practice_date).days == streak:
                    streak += 1
                else:
                    break
        
        # Get average accuracy
        cursor.execute('SELECT AVG(accuracy) FROM practice_sessions WHERE user_id = ?', (user_id,))
        avg_accuracy = cursor.fetchone()[0] or 0
        
        return jsonify({
            'words_practiced': words_practiced,
            'current_streak': streak,
            'accuracy': round(avg_accuracy, 1)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/words/random', methods=['GET'])
def get_random_word():
    difficulty = request.args.get('difficulty', 'beginner')
    
    try:
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT word, phonetic FROM words WHERE difficulty_level = ?', (difficulty,))
        words = cursor.fetchall()
        
        if not words:
            cursor.execute('SELECT word, phonetic FROM words')
            words = cursor.fetchall()
        
        if words:
            word_data = random.choice(words)
            audio_data = generate_audio(word_data[0])
            
            return jsonify({
                'word': word_data[0],
                'phonetic': word_data[1],
                'audio': audio_data
            }), 200
        else:
            return jsonify({'error': 'No words available'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/pronunciation/check', methods=['POST'])
def check_pronunciation():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    word = data.get('word')
    
    if not word:
        return jsonify({'error': 'Word is required'}), 400
    
    try:
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        # Get word phonetic
        cursor.execute('SELECT phonetic FROM words WHERE word = ?', (word.lower(),))
        result = cursor.fetchone()
        
        if result:
            phonetic = result[0]
            audio_data = generate_audio(word)
            
            return jsonify({
                'word': word,
                'phonetic': phonetic,
                'audio': audio_data,
                'tips': f'Focus on the pronunciation: {phonetic}. Practice each syllable slowly.'
            }), 200
        else:
            return jsonify({
                'word': word,
                'phonetic': '[...]',
                'audio': None,
                'tips': 'This word is not in our current database. We\'re constantly adding new words!'
            }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/pronunciation/analyze', methods=['POST'])
def analyze_pronunciation():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Get audio data from request
        audio_file = request.files.get('audio')
        reference_word = request.form.get('word')
        
        if not audio_file or not reference_word:
            return jsonify({'error': 'Audio file and reference word required'}), 400
        
        # Save and process audio
        audio_path = f"temp_audio_{session['user_id']}.wav"
        audio_file.save(audio_path)
        
        # Load and transcribe audio
        audio_data, sample_rate = sf.read(audio_path)
        user_transcription = transcribe_audio(audio_data, sample_rate)
        
        # Analyze pronunciation
        accuracy, feedback, ref_phonemes, user_phonemes = phonetic_comparison(reference_word, user_transcription)
        
        # Save to database
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        # Get word phonetic
        cursor.execute('SELECT phonetic FROM words WHERE word = ?', (reference_word.lower(),))
        result = cursor.fetchone()
        word_phonetic = result[0] if result else '/unknown/'
        
        cursor.execute('''
            INSERT INTO practice_sessions (user_id, word, phonetic, user_transcription, accuracy, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], reference_word, word_phonetic, user_transcription, accuracy, feedback))
        
        conn.commit()
        conn.close()
        
        # Clean up temp file
        os.remove(audio_path)
        
        return jsonify({
            'accuracy': round(accuracy, 1),
            'feedback': feedback,
            'user_transcription': user_transcription,
            'is_correct': accuracy >= 70
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_practice_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    try:
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        # Get practice history with pagination
        offset = (page - 1) * per_page
        cursor.execute('''
            SELECT word, phonetic, accuracy, feedback, session_date, user_transcription
            FROM practice_sessions 
            WHERE user_id = ?
            ORDER BY session_date DESC
            LIMIT ? OFFSET ?
        ''', (user_id, per_page, offset))
        
        sessions = cursor.fetchall()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM practice_sessions WHERE user_id = ?', (user_id,))
        total = cursor.fetchone()[0]
        
        history_data = []
        for session in sessions:
            history_data.append({
                'word': session[0],
                'phonetic': session[1],
                'accuracy': session[2],
                'feedback': session[3],
                'date': session[4],
                'user_transcription': session[5]
            })
        
        return jsonify({
            'history': history_data,
            'total': total,
            'page': page,
            'per_page': per_page
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/words/list', methods=['GET'])
def get_words_list():
    difficulty = request.args.get('difficulty')
    
    try:
        conn = sqlite3.connect('french_phonetics.db')
        cursor = conn.cursor()
        
        if difficulty:
            cursor.execute('SELECT word, phonetic, difficulty_level FROM words WHERE difficulty_level = ?', (difficulty,))
        else:
            cursor.execute('SELECT word, phonetic, difficulty_level FROM words')
        
        words = cursor.fetchall()
        
        words_list = []
        for word in words:
            words_list.append({
                'word': word[0],
                'phonetic': word[1],
                'difficulty': word[2]
            })
        
        return jsonify({'words': words_list}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
    populate_words_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
