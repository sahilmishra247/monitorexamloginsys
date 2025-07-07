# Use a slim Python base image for smaller size
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
# Since Render's Root Directory is 'testing_voice_fing', requirements.txt is directly accessible.
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
# This prevents it from trying to download at runtime, which can cause issues or delays
RUN python -c "from resemblyzer import VoiceEncoder; encoder = VoiceEncoder(); print('Resemblyzer model downloaded.')"

# Copy the application code
# Since Render's Root Directory is 'testing_voice_fing', app.py is directly accessible.
COPY app.py .

# Copy your static folder (assuming index.html is here)
# Important: Removed the inline comment as it caused a parsing error previously.
# Since Render's Root Directory is 'testing_voice_fing', static/ is directly accessible.
COPY static/ ./static/

# Expose the port that Render will use (Render sets PORT environment variable)
# Gunicorn will bind to this port
EXPOSE $PORT

# Command to run the application using Gunicorn
# 'app:app' refers to the 'app' object in 'app.py' (formerly authentication.py).
# It will bind to 0.0.0.0 on the port specified by the PORT environment variable.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]