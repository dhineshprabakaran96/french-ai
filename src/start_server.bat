@echo off
echo Starting French Pronunciation Assessment Server (Full AI Features)...
cd french-ai

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Flask server with AI pronunciation assessment...
python app.py
pause 