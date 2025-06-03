# French Pronunciation Assessment System

Integrated frontend and backend for French phonetics learning with AI-powered pronunciation assessment and user authentication.

## Quick Start

### Method 1: Automated Setup (Recommended)
```bash
# Run the setup script (Windows)
setup_venv.bat

# Start the server
start_server.bat
```

### Method 2: Manual Setup
```bash
# 1. Create and activate virtual environment
cd french-ai
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python app.py
```

### 3. Open Frontend
Open `french/index.html` in your browser or serve it locally:
```bash
cd french
python -m http.server 8000
# Then visit http://localhost:8000
```

## System Features

### Backend (`french-ai/app.py`)
- **AI Pronunciation Assessment** using Wav2Vec2 model
- **Real Phonetic Analysis** with eSpeak NG
- **User Authentication** with secure password hashing
- **Session Management** for logged-in users
- **Text-to-Speech** with Google TTS
- **Practice History Tracking** with statistics
- **REST API** for all frontend interactions

### Frontend (`french/`)
- **User Registration & Login** with validation
- **Real-time Audio Recording** for pronunciation practice
- **Progress Tracking Dashboard** with statistics
- **Practice History Viewer** with search and filtering
- **Phonetic Lookup** for any French word
- **Responsive Design** for mobile and desktop

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/user` - Get user info
- `POST /api/auth/logout` - User logout

### Words & Practice
- `GET /api/words/random` - Get random French word
- `GET /api/words/all` - Get all available words
- `POST /api/pronunciation/reference` - Generate reference audio
- `POST /api/pronunciation/assess` - Assess user pronunciation
- `POST /api/pronunciation/phonetic` - Get phonetic transcription
- `GET /api/practice/history` - Get practice history
- `GET /api/practice/stats` - Get practice statistics

## Requirements

### System Dependencies
- **Python 3.8+**
- **eSpeak NG** (for phonetic analysis)
  - Windows: Download from [eSpeak NG releases](https://github.com/espeak-ng/espeak-ng/releases)
  - Install to `C:\Program Files\eSpeak NG\`

### Python Dependencies
All listed in `french-ai/requirements.txt`:
- Flask & Flask-CORS
- PyTorch & Transformers
- Phonemizer & eSpeak integration
- Audio processing libraries
- Authentication utilities

## Usage Flow

1. **Setup**: Run `setup_venv.bat` to install dependencies
2. **Start Backend**: Run `start_server.bat` 
3. **Access Frontend**: Open `http://localhost:8000` in browser
4. **Register**: Create an account with any email domain
5. **Login**: Use your registered credentials
6. **Practice**: Record pronunciations and get AI feedback
7. **Track Progress**: View statistics and history

## Technologies

- **Backend**: Python Flask, PyTorch, Wav2Vec2, eSpeak NG
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **AI Model**: Facebook's Wav2Vec2 for French
- **Audio**: Web Audio API, Google Text-to-Speech
- **Authentication**: SHA-256 hashing, session tokens

## File Structure
```
├── french-ai/          # Backend API
│   ├── venv/          # Virtual environment
│   ├── app.py         # Flask server
│   ├── requirements.txt
│   └── frenchWordsDataset.csv
├── french/            # Frontend
│   ├── index.html     # Login page
│   ├── homepage.html  # Dashboard
│   ├── learnFrench.html # Practice interface
│   └── history.html   # Progress history
├── setup_venv.bat     # Setup script
├── start_server.bat   # Server launcher
└── README.md
``` 
