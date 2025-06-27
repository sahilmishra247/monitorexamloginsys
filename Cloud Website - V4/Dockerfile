# Dockerfile
FROM python:3.9-slim

# Install system dependencies required for OpenCV, dlib (used by face_recognition), and librosa
# libatlas-base-dev and gfortran are crucial for numpy/scipy and dlib
# libjpeg-dev and libpng-dev for image processing
# build-essential and cmake for building some Python packages from source (like dlib)
# pkg-config is often useful for building C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# IMPORTANT: Install setuptools and wheel BEFORE your requirements.txt
# This ensures the build backend for complex packages is available.
RUN pip install --no-cache-dir setuptools wheel

# Copy only requirements.txt first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your Flask app will run on. Render provides this via $PORT.
EXPOSE $PORT

# Command to run your application using Gunicorn
# 'authentication:app' means it's looking for the 'app' Flask instance
# inside the 'authentication.py' file. This is correct if your main
# Flask application file is indeed named authentication.py.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:$PORT", "authentication:app"]