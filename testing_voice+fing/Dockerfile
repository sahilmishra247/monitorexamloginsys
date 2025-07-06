# Use a slim Python base image for smaller size
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY authentication.py .
COPY static/ ./static/ # Copy your static folder (assuming index.html is here)

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application using Gunicorn
# 'authentication:app' refers to the 'app' object in 'authentication.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "authentication:app"]
