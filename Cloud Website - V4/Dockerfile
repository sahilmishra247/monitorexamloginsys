FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# NEW: Upgrade pip first to avoid dependency resolution issues
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# NEW: Install packages individually to avoid conflicts
RUN pip install --no-cache-dir \
    Flask==2.0.1 \
    Flask-Cors==3.0.10 \
    numpy==1.21.2 \
    librosa==0.9.1 \
    resemblyzer==0.1.3 \
    opencv-contrib-python-headless==4.5.3.56 \
    mysql-connector-python==8.0.26 \
    python-dotenv==0.19.0 \
    scipy==1.7.1 \
    soundfile==0.10.3.post1

COPY . .

RUN mkdir -p /app/monitorexamloginsys/embeddings /app/static

ENV FLASK_APP=authentication.py
ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "authentication:app"]