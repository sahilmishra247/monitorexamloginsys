# Use a slim Python base image for smaller size
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
# This path is relative to the Docker Build Context Directory (testing_voice_fing/)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for librosa (libsndfile is often needed) and ffmpeg
# ffmpeg is useful for handling various audio formats if librosa needs it
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Force resemblyzer model download during build
RUN python -c "from resemblyzer import VoiceEncoder; encoder = VoiceEncoder(); print('Resemblyzer model downloaded.')"

# Copy the application code
# This path is relative to the Docker Build Context Directory (testing_voice_fing/)
COPY app.py .

# Copy your static folder
# This path is relative to the Docker Build Context Directory (testing_voice_fing/)
COPY static/ ./static/

# Expose the port that Render will use (Render sets PORT environment variable)
EXPOSE $PORT

# Command to run the application using Gunicorn
# 'app:app' refers to the the 'app' object in 'app.py' which is now copied to /app in the container
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]