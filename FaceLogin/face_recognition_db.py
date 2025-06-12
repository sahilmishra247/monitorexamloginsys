from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
import numpy as np
import os
import base64
import io
import json
from datetime import datetime
import cv2 # Import OpenCV

app = Flask(__name__)
CORS(app)

# Directory to store the registered face encodings
FACES_DIR = "registered_faces"
TOLERANCE = 0.6  # Lower tolerance means stricter match

# Ensure the registered_faces directory exists
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
    print(f"Created directory: {FACES_DIR}")

def load_known_faces():
    """
    Loads all known face encodings and their corresponding usernames from the FACES_DIR.
    Returns two lists: known_face_encodings and known_face_names.
    """
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(FACES_DIR):
        if filename.endswith('.npy'):
            username = os.path.splitext(filename)[0]
            encoding_path = os.path.join(FACES_DIR, filename)
            try:
                face_encoding = np.load(encoding_path)
                known_face_encodings.append(face_encoding)
                known_face_names.append(username)
            except Exception as e:
                print(f"Error loading encoding for {username}: {e}")
                
    return known_face_encodings, known_face_names

def save_face_data(image_data, username):
    """
    Save face image and encoding from base64 image data using OpenCV.
    """
    try:
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array, then to OpenCV image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Read as BGR
        
        if image_cv is None:
            return False, "Could not decode image bytes."

        # Convert BGR (OpenCV default) to RGB (face_recognition default)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image_rgb)
        
        if len(face_locations) == 0:
            return False, "No face detected in the image"
        elif len(face_locations) > 1:
            return False, "Multiple faces detected. Please ensure only one person is in the frame"
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        if len(face_encodings) == 0:
            return False, "Could not generate face encoding"
        
        face_encoding = face_encodings[0]
        
        # Save the encoding as .npy file
        encoding_path = os.path.join(FACES_DIR, f"{username}.npy")
        np.save(encoding_path, face_encoding)
        
        # Save the original image as .jpg file using OpenCV (optional, for debugging/visualisation)
        image_path = os.path.join(FACES_DIR, f"{username}.jpg")
        cv2.imwrite(image_path, image_cv) # Save in BGR format
        
        print(f"Successfully saved face encoding and image for '{username}'")
        return True, "Face registered successfully"
        
    except Exception as e:
        print(f"Error saving face data: {str(e)}")
        return False, f"Error processing image: {str(e)}"

@app.route('/')
def index():
    """Serve the registration page"""
    try:
        return send_from_directory('.', 'register.html')
    except:
        return """
        <h1>Registration Page</h1>
        <p>Please make sure 'register.html' exists in the same directory as this script.</p>
        <p><a href="/simple">Try Simple Registration Page</a></p>
        <p><a href="/login">Go to Login Page</a></p>
        """

@app.route('/simple')
def simple_page():
    """Simple registration page"""
    # This HTML remains the same as the client-side still sends base64
    return '''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Simple Face Registration</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f4f7f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .container {
            background: #ffffff;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 400px;
        }

        h1 {
            margin-bottom: 20px;
            color: #333;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px 15px;
            margin-bottom: 15px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 8px;
            outline: none;
            transition: border 0.3s ease;
        }

        input[type="text"]:focus {
            border-color: #007BFF;
        }

        video, canvas {
            border-radius: 8px;
            border: 1px solid #ccc;
            margin: 10px auto;
            display: block;
            width: 100%;
            max-width: 320px;
        }

        button {
            padding: 10px 20px;
            margin: 8px 4px;
            font-size: 15px;
            background-color: #007BFF;
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }

        button:hover:not(:disabled) {
            background-color: #0056b3;
        }

        .message {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
            display: inline-block;
        }

        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Registration</h1>
        <input type="text" id="username" placeholder="Enter your name" />
        <video id="video" width="320" height="240" autoplay></video>
        <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
        <div>
            <button onclick="startCamera()">Start Camera</button>
            <button onclick="capturePhoto()" id="captureBtn" disabled>Capture Photo</button>
            <button onclick="registerUser()" id="registerBtn" disabled>Register User</button>
        </div>
        <div id="message"></div>
    </div>

    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let capturedImage = null;

        function showMessage(text, type) {
            document.getElementById('message').innerHTML = 
                '<div class="message ' + type + '">' + text + '</div>';
        }

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                document.getElementById('captureBtn').disabled = false;
                showMessage('Camera started successfully!', 'success');
            } catch (err) {
                showMessage('Error accessing camera: ' + err.message, 'error');
            }
        }

        function capturePhoto() {
            const username = document.getElementById('username').value.trim();
            if (!username) {
                showMessage('Please enter your name first!', 'error');
                return;
            }

            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);

            capturedImage = canvas.toDataURL('image/jpeg', 0.8);
            canvas.style.display = 'block';
            document.getElementById('registerBtn').disabled = false;
            showMessage('Photo captured! Now click Register User.', 'success');
        }

        async function registerUser() {
            const username = document.getElementById('username').value.trim();

            if (!username || !capturedImage) {
                showMessage('Please enter name and capture photo first!', 'error');
                return;
            }

            try {
                showMessage('Registering user...', 'success');

                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: username,
                        image_data: capturedImage
                    })
                });

                const result = await response.json();

                if (result.success) {
                    showMessage('Registration successful! User ' + username + ' has been registered.', 'success');
                    document.getElementById('username').value = '';
                    canvas.style.display = 'none';
                    capturedImage = null;
                    document.getElementById('registerBtn').disabled = true;
                } else {
                    showMessage('Registration failed: ' + result.message, 'error');
                }
            } catch (error) {
                showMessage('Error: ' + error.message, 'error');
            }
        }

        window.onload = () => startCamera();
    </script>
</body>
</html>

    '''

@app.route('/login')
def login_page():
    """Serve the login page"""
    try:
        return send_from_directory('.', 'login.html')
    except Exception as e:
        return f"""
        <h1>Login Page</h1>
        <p>Please make sure 'login.html' exists in the same directory as this script.</p>
        <p>Error: {str(e)}</p>
        <p><a href="/simple">Go to Registration Page</a></p>
        """

@app.route('/api/register', methods=['POST'])
def register_face():
    """API endpoint to register a new face"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
        
        username = data.get('username', '').strip()
        image_data = data.get('image_data', '')
        
        if not username:
            return jsonify({'success': False, 'message': 'Username is required'}), 400
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image data is required'}), 400
        
        # Check if username already exists
        encoding_file = os.path.join(FACES_DIR, f"{username}.npy")
        if os.path.exists(encoding_file):
            return jsonify({'success': False, 'message': 'Username already exists. Please choose a different username or login.'}), 400
        
        # Save face image and encoding
        success, message = save_face_data(image_data, username)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        print(f"Error in register_face: {str(e)}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/login', methods=['POST'])
def login_face():
    """API endpoint to login user by face recognition using OpenCV."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
            
        image_data = data.get('image_data', '')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image data is required'}), 400
            
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array, then to OpenCV image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Read as BGR
        
        if image_cv is None:
            return jsonify({'success': False, 'message': 'Could not decode image from provided data.'}), 400

        # Convert BGR (OpenCV default) to RGB (face_recognition default)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings in the captured image
        face_locations = face_recognition.face_locations(image_rgb) # Use RGB image here
        
        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image. Please try again.'}), 400
        elif len(face_locations) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one person is in the frame.'}), 400
            
        captured_face_encoding = face_recognition.face_encodings(image_rgb, face_locations)[0] # Use RGB image here
        
        # Load known faces from directory
        known_face_encodings, known_face_names = load_known_faces()

        if not known_face_encodings:
            return jsonify({'success': False, 'message': 'No registered users found. Please register first.'})

        # Compare the captured face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, captured_face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_face_encodings, captured_face_encoding)
        
        # Find the best match
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            username = known_face_names[best_match_index]
            return jsonify({'success': True, 'message': f'Login successful! Welcome, {username}!', 'username': username})
        else:
            return jsonify({'success': False, 'message': 'Face not recognized. Please try again or register.'}), 401
            
    except Exception as e:
        print(f"Error in login_face: {str(e)}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/users', methods=['GET'])
def get_users():
    """API endpoint to get all registered users"""
    try:
        users = []
        for filename in os.listdir(FACES_DIR):
            if filename.endswith('.npy'):
                username = os.path.splitext(filename)[0]
                filepath = os.path.join(FACES_DIR, filename)
                timestamp = os.path.getctime(filepath)
                
                users.append({
                    'username': username,
                    'registered_at': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by registration time (newest first)
        users.sort(key=lambda x: x['registered_at'], reverse=True)
        
        return jsonify({'success': True, 'users': users})
        
    except Exception as e:
        print(f"Error getting users: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def server_status():
    """API endpoint to check server status"""
    try:
        registered_count = len([f for f in os.listdir(FACES_DIR) if f.endswith('.npy')])
        return jsonify({
            'status': 'online',
            'faces_directory': FACES_DIR,
            'registered_users': registered_count,
            'tolerance': TOLERANCE
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("Face Recognition Server Starting...")
    print(f"Faces will be saved to: {os.path.abspath(FACES_DIR)}")
    print("Server will be available at: http://localhost:5000")
    print("Simple registration page: http://localhost:5000/simple")
    print("Login page: http://localhost:5000/login")
    print("="*50)
    
    # Run the server
    app.run(debug=True, host='127.0.0.1', port=5000)