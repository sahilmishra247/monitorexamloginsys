# Stage 1: Build environment
FROM python:3.9.18-slim as builder

# Install system dependencies
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.9.18-slim

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

# Copy installed packages
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser . .

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PORT=8000 \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose and run
EXPOSE $PORT
CMD ["sh", "-c", "gunicorn --workers ${WORKERS:-4} --bind 0.0.0.0:${PORT} authentication:app"]