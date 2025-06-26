# authentication.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import numpy as np
import os
import librosa
from resemblyzer import VoiceEncoder
import io
import face_recognition
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from datetime import datetime
import cv2

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    SAMPLE_RATE = 16000
    VOICE_THRESHOLD = 0.75
    EMBEDDING_DIR = "monitorexamloginsys/embeddings"
    STATIC_DIR = "static"
    
    # Future thresholds for other biometric methods
    FACE_THRESHOLD = 0.90
    FINGERPRINT_THRESHOLD = 200  # Adjusted to match app.py's threshold of 200 matches

# Ensure directories exist
os.makedirs(Config.EMBEDDING_DIR, exist_ok=True)
os.makedirs(Config.STATIC_DIR, exist_ok=True)

# Abstract base class for biometric authentication
class BiometricAuthenticator(ABC):
    def __init__(self, auth_type: str, threshold: float):
        self.auth_type = auth_type
        self.threshold = threshold
        self.data_dir = os.path.join(Config.EMBEDDING_DIR, auth_type)
        os.makedirs(self.data_dir, exist_ok=True)
    
    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        """Extract features from raw biometric data"""
        pass
    
    @abstractmethod
    def preprocess_data(self, data: Any) -> Any:
        """Preprocess the biometric data"""
        pass
        
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        a = np.array(features1)
        b = np.array(features2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def save_features(self, user_id: str, features: np.ndarray) -> bool:
        """Save user features to file"""
        try:
            feature_path = os.path.join(self.data_dir, f"{user_id}.npy")
            np.save(feature_path, features)
            return True
        except Exception as e:
            print(f"Error saving features: {e}")
            return False
    
    def load_features(self, user_id: str) -> Optional[np.ndarray]:
        """Load user features from file"""
        try:
            feature_path = os.path.join(self.data_dir, f"{user_id}.npy")
            print(f"Loading features from: {feature_path}")
            if os.path.exists(feature_path):
                return np.load(feature_path)
            return None
        except Exception as e:
            print(f"Error loading features: {e}")
            return None
    
    def register_user(self, user_id: str, data: Any) -> Dict[str, Any]:
        """Register a new user"""
        print(f"Registering user '{user_id}' for {self.auth_type}")
        # Check if user already exists
        if self.load_features(user_id) is not None:
            return {
                "success": False,
                "message": f"User '{user_id}' already registered for {self.auth_type}.",
                "auth_type": self.auth_type
            }
        
        try:
            # Process data and extract features
            processed_data = self.preprocess_data(data)
            if isinstance(processed_data, tuple):
                image_data, extension = processed_data
                features = self.extract_features(image_data)
                features['extension'] = extension
            else:
                features = self.extract_features(processed_data)
            
            # Save features
            if self.save_features(user_id, features):
                return {
                    "success": True,
                    "message": f"Successfully registered '{user_id}' for {self.auth_type}.",
                    "auth_type": self.auth_type
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to save {self.auth_type} data for '{user_id}'.",
                    "auth_type": self.auth_type
                }
        except Exception as e:
            print(f"Error in register_user: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing {self.auth_type} data: {str(e)}",
                "auth_type": self.auth_type
            }
    
    def authenticate_user(self, user_id: str, data: Any) -> Dict[str, Any]:
        """Authenticate an existing user"""
        print(f"Authenticating user '{user_id}' for {self.auth_type}")
        # Load stored features
        stored_features = self.load_features(user_id)
        if stored_features is None:
            return {
                "success": False,
                "message": f"No registered {self.auth_type} data found for user '{user_id}'.",
                "auth_type": self.auth_type,
                "similarity": 0.0
            }
        
        try:
            # Process data and extract features
            processed_data = self.preprocess_data(data)
            if isinstance(processed_data, tuple):
                image_data, extension = processed_data
                test_features = self.extract_features(image_data)
            else:
                test_features = self.extract_features(processed_data)
            
            # Calculate similarity
            similarity = self.calculate_similarity(test_features, stored_features)
            
            if similarity >= self.threshold:
                return {
                    "success": True,
                    "message": f"Authentication successful for '{user_id}' using {self.auth_type}.",
                    "auth_type": self.auth_type,
                    "similarity": round(similarity, 3)
                }
            else:
                return {
                    "success": False,
                    "message": f"Authentication failed for '{user_id}' using {self.auth_type}.",
                    "auth_type": self.auth_type,
                    "similarity": round(similarity, 3)
                }
        except Exception as e:
            print(f"Error in authenticate_user: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing {self.auth_type} authentication: {str(e)}",
                "auth_type": self.auth_type,
                "similarity": 0.0
            }

# Voice Authentication Implementation
class VoiceAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("voice", Config.VOICE_THRESHOLD)
        self.encoder = VoiceEncoder()
    
    def preprocess_data(self, data: bytes) -> np.ndarray:
        """Preprocess audio data"""
        print("Preprocessing voice data")
        y, _ = librosa.load(io.BytesIO(data), sr=Config.SAMPLE_RATE)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed / np.max(np.abs(y_trimmed)) if np.max(np.abs(y_trimmed)) > 0 else y_trimmed
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract voice features using Resemblyzer"""
        print("Extracting voice features")
        return self.encoder.embed_utterance(data)

# Updated Fingerprint Authentication Implementation
class FingerprintAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("fingerprint", Config.FINGERPRINT_THRESHOLD)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.allowed_extensions = {'.jpg', '.jpeg', '.png'}  # Define allowed file extensions
    
    def get_file_extension(self, filename: str) -> str:
        """Extract the file extension from the filename"""
        return os.path.splitext(filename)[1].lower()
    
    def preprocess_data(self, fingerprint_file: Any) -> np.ndarray:
        """Preprocess fingerprint image data by reading the uploaded file"""
        try:
            # Validate file extension
            extension = self.get_file_extension(fingerprint_file.filename)
            if extension not in self.allowed_extensions:
                raise ValueError(f"Unsupported file format: {extension}. Use JPG or PNG.")
            
            print(f"Processing file: {fingerprint_file.filename}")
            # Read the file directly into a numpy array
            nparr = np.frombuffer(fingerprint_file.read(), np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError("Could not decode fingerprint image")
            
            return image, extension
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise ValueError(f"Error preprocessing fingerprint image: {str(e)}")
    
    def extract_features(self, image_data: np.ndarray) -> dict:
        """Extract fingerprint features using ORB detector"""
        try:
            print("Extracting fingerprint features")
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.orb.detectAndCompute(image_data, None)
            
            if descriptors is None or len(descriptors) == 0:
                raise ValueError("No keypoints detected in fingerprint image")
            
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image': image_data  # Store the image for saving
            }
        except Exception as e:
            print(f"Error extracting fingerprint features: {str(e)}")
            raise ValueError(f"Error extracting fingerprint features: {str(e)}")
    
    def calculate_similarity(self, features1: dict, features2: dict) -> float:
        """Calculate similarity between two fingerprints using ORB matching"""
        try:
            print("Calculating fingerprint similarity")
            des1 = features1['descriptors']
            des2 = features2['descriptors']
            
            if des1 is None or des2 is None:
                return 0.0
            
            # Match descriptors
            matches = self.bf.match(des1, des2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Return the number of matches as the similarity score
            return float(len(matches))
        except Exception as e:
            print(f"Error calculating fingerprint similarity: {e}")
            return 0.0
    
    def save_features(self, user_id: str, features: dict) -> bool:
        """Save the fingerprint image with the original extension in the embeddings folder"""
        try:
            # Use the extension stored in features
            extension = features.get('extension', '.jpg')  # Default to .jpg if not specified
            image_path = os.path.join(self.data_dir, f"{user_id}{extension}")
            print(f"Saving fingerprint image to: {image_path}")
            cv2.imwrite(image_path, features['image'])
            return True
        except Exception as e:
            print(f"Error saving fingerprint image: {e}")
            return False
    
    def load_features(self, user_id: str) -> Optional[dict]:
        """Load the fingerprint image and extract features"""
        try:
            # Try loading the image with any of the allowed extensions
            for ext in self.allowed_extensions:
                image_path = os.path.join(self.data_dir, f"{user_id}{ext}")
                if os.path.exists(image_path):
                    print(f"Loading fingerprint image from: {image_path}")
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Could not load fingerprint image for user '{user_id}'")
                        return None
                    features = self.extract_features(image)
                    features['extension'] = ext  # Store the extension for saving
                    return features
            print(f"No fingerprint image found for user '{user_id}'")
            return None
        except Exception as e:
            print(f"Error loading fingerprint image: {e}")
            return None

# Placeholder class for future face authentication
class FaceAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("face", Config.FACE_THRESHOLD)
    
    def preprocess_data(self, image_bytes: bytes) -> Any:
        # Load image from bytes
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        return image  # Return the loaded image for encoding
    
    def extract_features(self, image_data: Any) -> np.ndarray:
        # Get face encodings (features)
        encodings = face_recognition.face_encodings(image_data)
        if encodings:
            return encodings[0]  # Use the first encoding if multiple faces are detected
        else:
            return np.array([])  # Return empty array if no faces found

# Biometric Manager
class BiometricManager:
    def __init__(self):
        self.authenticators = {
            "voice": VoiceAuthenticator(),
            "face": FaceAuthenticator(),
            "fingerprint": FingerprintAuthenticator()
        }
    
    def get_authenticator(self, auth_type: str) -> Optional[BiometricAuthenticator]:
        return self.authenticators.get(auth_type)
    
    def get_available_methods(self) -> list:
        """Return list of available authentication methods"""
        available = []
        for method, authenticator in self.authenticators.items():
            try:
                if method == "voice":
                    available.append(method)
                elif method == "face":
                    available.append(method)  # Add when implemented
                    pass
                elif method == "fingerprint":
                    available.append(method)
            except NotImplementedError:
                pass
        return available

# Initialize biometric manager
biometric_manager = BiometricManager()

# Helper functions
def decode_base64(b64_string: str) -> bytes:
    """Decode base64 string to bytes"""
    try:
        return base64.b64decode(b64_string)
    except Exception:
        raise ValueError("Invalid base64 string")

def validate_request_data(data: dict, required_fields: list) -> tuple:
    """Validate request data"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, ""

# API Routes
@app.route('/')
def serve_static():
    """Serve static files"""
    print("Serving static file: index.html")
    return send_from_directory(Config.STATIC_DIR, 'index.html')

@app.route('/static/<path:path>')
def serve_static_files(path):
    print(f"Serving static file: {path}")
    return send_from_directory(Config.STATIC_DIR, path)

@app.route('/api/methods', methods=['GET'])
def get_available_methods():
    """Get available authentication methods"""
    print("Handling /api/methods request")
    return jsonify({
        "available_methods": biometric_manager.get_available_methods(),
        "all_methods": list(biometric_manager.authenticators.keys())
    })

@app.route('/api/register', methods=['POST'])
def register():
    """Register user with biometric data"""
    print("Received /api/register request")
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Map frontend field names to backend names
        user_id = data.get('username') or data.get('user_id')
        auth_type = data.get('auth_method') or data.get('auth_type')
        
        # Get the appropriate data
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing voice biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'face':
            data_b64 = data.get('face_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing face biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'fingerprint':
            if 'fingerprint' not in request.files:
                return jsonify({
                    "success": False,
                    "message": "Missing fingerprint file"
                }), 400
            fingerprint_file = request.files['fingerprint']
            if not fingerprint_file.filename:
                return jsonify({
                    "success": False,
                    "message": "No fingerprint file selected"
                }), 400
            # Validate file extension
            extension = os.path.splitext(fingerprint_file.filename)[1].lower()
            if extension not in {'.jpg', '.jpeg', '.png'}:
                return jsonify({
                    "success": False,
                    "message": f"Unsupported file format: {extension}. Use JPG or PNG."
                }), 400
            raw_data = fingerprint_file
        else:
            return jsonify({
                "success": False,
                "message": "No biometric data provided"
            }), 400
        
        # Validate required fields
        if not all([user_id, auth_type]):
            missing = []
            if not user_id: missing.append('username/user_id')
            if not auth_type: missing.append('auth_method/auth_type')
            return jsonify({
                "success": False,
                "message": f"Missing required fields: {', '.join(missing)}"
            }), 400
        
        # Get authenticator
        authenticator = biometric_manager.get_authenticator(auth_type)
        if not authenticator:
            return jsonify({
                "success": False,
                "message": f"Unsupported authentication type: {auth_type}"
            }), 400
        
        # Register user
        result = authenticator.register_user(user_id, raw_data)
        
        print(f"Register response: {result}")
        status_code = 200 if result["success"] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        print(f"Error in /api/register: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate user with biometric data"""
    print("Received /api/login request")
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Map frontend field names to backend names
        user_id = data.get('username') or data.get('user_id')
        auth_type = data.get('auth_method') or data.get('auth_type')
        
        print(f"Login attempt: {user_id} using {auth_type}")
        
        # Get the appropriate data
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing voice biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'face':
            data_b64 = data.get('face_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing face biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'fingerprint':
            if 'fingerprint' not in request.files:
                return jsonify({
                    "success": False,
                    "message": "Missing fingerprint file"
                }), 400
            fingerprint_file = request.files['fingerprint']
            if not fingerprint_file.filename:
                return jsonify({
                    "success": False,
                    "message": "No fingerprint file selected"
                }), 400
            # Validate file extension
            extension = os.path.splitext(fingerprint_file.filename)[1].lower()
            if extension not in {'.jpg', '.jpeg', '.png'}:
                return jsonify({
                    "success": False,
                    "message": f"Unsupported file format: {extension}. Use JPG or PNG."
                }), 400
            raw_data = fingerprint_file
        else:
            return jsonify({
                "success": False,
                "message": "No biometric data provided"
            }), 400
        
        # Validate required fields
        if not all([user_id, auth_type]):
            missing = []
            if not user_id: missing.append('username/user_id')
            if not auth_type: missing.append('auth_method/auth_type')
            return jsonify({
                "success": False,
                "message": f"Missing required fields: {', '.join(missing)}"
            }), 400
        
        # Get authenticator
        authenticator = biometric_manager.get_authenticator(auth_type)
        if not authenticator:
            return jsonify({
                "success": False,
                "message": f"Unsupported authentication type: {auth_type}"
            }), 400
        
        # Authenticate user
        result = authenticator.authenticate_user(user_id, raw_data)
        
        print(f"Login response: {result}")
        status_code = 200 if result["success"] else 401
        return jsonify(result), status_code
        
    except Exception as e:
        print(f"Error in /api/login: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/user/<user_id>/methods', methods=['GET'])
def get_user_registered_methods(user_id):
    """Get authentication methods registered for a specific user"""
    print(f"Received /api/user/{user_id}/methods request")
    registered_methods = []
    
    for method_name, authenticator in biometric_manager.authenticators.items():
        if authenticator.load_features(user_id) is not None:
            registered_methods.append(method_name)
    
    return jsonify({
        "user_id": user_id,
        "registered_methods": registered_methods
    })

@app.route('/api/user/<user_id>/delete', methods=['DELETE'])
def delete_user_data(user_id):
    """Delete all biometric data for a user"""
    print(f"Received /api/user/{user_id}/delete request")
    deleted_methods = []
    errors = []
    
    for method_name, authenticator in biometric_manager.authenticators.items():
        try:
            feature_path = os.path.join(authenticator.data_dir, f"{user_id}.npy")
            if os.path.exists(feature_path):
                os.remove(feature_path)
                deleted_methods.append(method_name)
            # For fingerprint, we save as .jpg or .png
            if method_name == "fingerprint":
                for ext in {'.jpg', '.jpeg', '.png'}:
                    image_path = os.path.join(authenticator.data_dir, f"{user_id}{ext}")
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        if method_name not in deleted_methods:
                            deleted_methods.append(method_name)
        except Exception as e:
            errors.append(f"Error deleting {method_name} data: {str(e)}")
    
    if errors:
        return jsonify({
            "success": False,
            "message": "Some data could not be deleted",
            "deleted_methods": deleted_methods,
            "errors": errors
        }), 500
    
    return jsonify({
        "success": True,
        "message": f"All biometric data deleted for user '{user_id}'",
        "deleted_methods": deleted_methods
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    print("404 error occurred")
    return jsonify({"success": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    print("500 error occurred")
    return jsonify({"success": False, "message": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available authentication methods:", biometric_manager.get_available_methods())
    app.run(debug=True, host='0.0.0.0', port=5000)
