"""
Face Recognition Authentication Module
This module only runs when called by the main server
"""

import face_recognition
import numpy as np
import os
import base64
import cv2
from datetime import datetime
from flask import request

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

def authenticate(request_obj):
    """
    Main authentication function called by the server
    This function only executes when user clicks face login
    """
    print("Face recognition module loaded and running...")
    
    try:
        # Handle different request types
        if request_obj.method == 'POST':
            # Check if it's JSON data (login) or form data (registration)
            if request_obj.is_json:
                data = request_obj.get_json()
                
                # Check if this is a registration request
                if 'username' in data and 'image_data' in data:
                    return handle_registration(data)
                # Otherwise it's a login request
                else:
                    return handle_login(data)
            else:
                return {"error": "Invalid request format"}
        else:
            return {"error": "Only POST requests are supported"}
            
    except Exception as e:
        print(f"Face recognition error: {e}")
        return {"error": f"Face recognition failed: {str(e)}"}

def handle_registration(data):
    """Handle face registration"""
    try:
        username = data.get('username', '').strip()
        image_data = data.get('image_data', '')
        
        if not username:
            return {'success': False, 'message': 'Username is required'}
        
        if not image_data:
            return {'success': False, 'message': 'Image data is required'}
        
        # Check if username already exists
        encoding_file = os.path.join(FACES_DIR, f"{username}.npy")
        if os.path.exists(encoding_file):
            return {'success': False, 'message': 'Username already exists. Please choose a different username or login.'}
        
        # Save face image and encoding
        success, message = save_face_data(image_data, username)
        
        if success:
            return {'success': True, 'message': message, 'redirect': '/success'}
        else:
            return {'success': False, 'message': message}
            
    except Exception as e:
        print(f"Error in handle_registration: {str(e)}")
        return {'success': False, 'message': f'Registration error: {str(e)}'}

def handle_login(data):
    """Handle face login using OpenCV"""
    try:
        image_data = data.get('image_data', '')
        
        if not image_data:
            return {'success': False, 'message': 'Image data is required'}
            
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array, then to OpenCV image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Read as BGR
        
        if image_cv is None:
            return {'success': False, 'message': 'Could not decode image from provided data.'}

        # Convert BGR (OpenCV default) to RGB (face_recognition default)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings in the captured image
        face_locations = face_recognition.face_locations(image_rgb) # Use RGB image here
        
        if len(face_locations) == 0:
            return {'success': False, 'message': 'No face detected in the image. Please try again.'}
        elif len(face_locations) > 1:
            return {'success': False, 'message': 'Multiple faces detected. Please ensure only one person is in the frame.'}
            
        captured_face_encoding = face_recognition.face_encodings(image_rgb, face_locations)[0] # Use RGB image here
        
        # Load known faces from directory
        known_face_encodings, known_face_names = load_known_faces()

        if not known_face_encodings:
            return {'success': False, 'message': 'No registered users found. Please register first.'}

        # Compare the captured face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, captured_face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_face_encodings, captured_face_encoding)
        
        # Find the best match
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            username = known_face_names[best_match_index]
            return {
                'success': True, 
                'message': f'Login successful! Welcome, {username}!', 
                'username': username,
                'redirect': '/success'
            }
        else:
            return {'success': False, 'message': 'Face not recognized. Please try again or register.'}
            
    except Exception as e:
        print(f"Error in handle_login: {str(e)}")
        return {'success': False, 'message': f'Login error: {str(e)}'}

def get_users():
    """Get all registered users"""
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
        
        return {'success': True, 'users': users}
        
    except Exception as e:
        print(f"Error getting users: {str(e)}")
        return {'success': False, 'message': str(e)}

def get_status():
    """Get server status"""
    try:
        registered_count = len([f for f in os.listdir(FACES_DIR) if f.endswith('.npy')])
        return {
            'status': 'online',
            'faces_directory': FACES_DIR,
            'registered_users': registered_count,
            'tolerance': TOLERANCE
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# This code only runs when the module is imported/called
print("Face login module initialized (but not executing until called)")