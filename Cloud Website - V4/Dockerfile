# Use a prebuilt image that includes dlib + face_recognition + OpenCV
FROM facefusion/face-recognition:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies for audio + image processing and HTTPS
RUN apt-get update && apt-get install -y \
    ffmpeg \               
    libsndfile1 \          
    libgl1 \               
    libsm6 \               
    libxext6 \             
    libxrender-dev \       
    ca-certificates \      
    git \                  
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code into the container
COPY . .

# Expose the port your Flask app uses (match with Gunicorn bind)
EXPOSE 5000

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
