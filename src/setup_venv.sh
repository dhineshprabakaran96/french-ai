#!/bin/bash

echo "Setting up French Pronunciation Assessment System..."
echo

cd french-ai

echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    echo "Make sure Python 3 is installed"
    exit 1
fi

echo
echo "Activating virtual environment..."
source venv/bin/activate

echo
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "Setup completed successfully!"
echo "To start the server:"
echo "1. cd french-ai"
echo "2. source venv/bin/activate"
echo "3. python app.py"
echo 