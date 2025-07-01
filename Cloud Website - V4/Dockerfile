# Use a standard Python image instead of one with dlib/face_recognition
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies for audio processing, OpenCV, and HTTPS
RUN apt-get update && apt-get install -y \
    ffmpeg \                  
    libsndfile1 \             
    libgl1 \                 
    libsm6 \ 
    libxext6 \ 
    libxrender-dev \  
    ca-certificates \        
    git \     
    build-essential \               
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN ls -al && echo "In working dir"
RUN cat requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Flask/Gunicorn
EXPOSE 5000

# Start app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
