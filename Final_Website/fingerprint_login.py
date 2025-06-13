import cv2
import os
import numpy as np
from flask import request

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for registered fingerprints
stored_images = {}

def get_fingerprint_input(file_storage):
    """Extract fingerprint image from uploaded file"""
    image_path = os.path.join(UPLOAD_FOLDER, file_storage.filename)
    file_storage.save(image_path)
    image = cv2.imread(image_path, 0)
    if image is None:
        print("Could not open or find the image.")
        return None
    return image

def compare_fingerprints(img1, img2):
    """Compare two fingerprint images using ORB feature matching"""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return False
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return len(matches) > 200

def authenticate(flask_request):
    """
    Main authentication function called by the Flask app
    Expected to receive form data with:
    - username: string
    - action: 'register' or 'login'
    - fingerprint: uploaded image file
    """
    try:
        username = flask_request.form.get("username")
        action = flask_request.form.get("action")
        file = flask_request.files.get("fingerprint")
        
        if not username:
            return {"success": False, "message": "Username is required."}
        
        if not file:
            return {"success": False, "message": "No fingerprint file uploaded."}
        
        # Process the fingerprint image
        image = get_fingerprint_input(file)
        if image is None:
            return {"success": False, "message": "Could not read fingerprint image."}
        
        if action == "register":
            # Register new fingerprint
            stored_images[username] = image
            return {
                "success": True, 
                "message": f"User '{username}' registered successfully.",
                "action": "register"
            }
            
        elif action == "login":
            # Authenticate user
            stored_image = stored_images.get(username)
            if stored_image is not None and compare_fingerprints(image, stored_image):
                return {
                    "success": True, 
                    "message": f"Login successful for user '{username}'.",
                    "action": "login",
                    "redirect": "/success"
                }
            else:
                return {
                    "success": False, 
                    "message": f"Login failed for user '{username}'. Please check your credentials.",
                    "action": "login"
                }
        
        else:
            return {"success": False, "message": "Invalid action specified."}
            
    except Exception as e:
        return {"success": False, "message": f"Authentication error: {str(e)}"}