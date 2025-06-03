# Use official Python slim image
ARG http_proxy=http://internet.ford.com:83
ARG https_proxy=http://internet.ford.com:83

FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project files
COPY . .

# Expose port 8080, Cloud Run requires this
ENV PORT=8080
EXPOSE 8080

# Run the app with Gunicorn server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
