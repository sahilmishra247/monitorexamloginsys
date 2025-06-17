# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import numpy as np
import os
import librosa
from resemblyzer import VoiceEncoder
import io
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    SAMPLE_RATE = 16000
    VOICE_THRESHOLD = 0.85
    EMBEDDING_DIR = "embeddings"
    STATIC_DIR = "static"
    
    # Future thresholds for other biometric methods
    FACE_THRESHOLD = 0.90
    FINGERPRINT_THRESHOLD = 0.95

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
    def extract_features(self, data_bytes: np.ndarray) -> np.ndarray:
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
            if os.path.exists(feature_path):
                return np.load(feature_path)
            return None
        except Exception as e:
            print(f"Error loading features: {e}")
            return None
    
    def register_user(self, user_id: str, data: bytes) -> Dict[str, Any]:
        """Register a new user"""
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
            return {
                "success": False,
                "message": f"Error processing {self.auth_type} data: {str(e)}",
                "auth_type": self.auth_type
            }
    
    def authenticate_user(self, user_id: str, data: bytes) -> Dict[str, Any]:
        """Authenticate an existing user"""
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
        y, _ = librosa.load(io.BytesIO(data), sr=Config.SAMPLE_RATE)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed / np.max(np.abs(y_trimmed)) if np.max(np.abs(y_trimmed)) > 0 else y_trimmed
    
    def extract_features(self, data_bytes: np.ndarray) -> np.ndarray:
        """Extract voice features using Resemblyzer"""
        return self.encoder.embed_utterance(data_bytes)

# Placeholder classes for future biometric methods
class FaceAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("face", Config.FACE_THRESHOLD)
    
    def preprocess_data(self, image_bytes: bytes) -> Any:
        # TODO: Implement face preprocessing
        raise NotImplementedError("Face authentication not yet implemented")
    
    def extract_features(self, image_data: Any) -> np.ndarray:
        # TODO: Implement face feature extraction
        raise NotImplementedError("Face authentication not yet implemented")

class FingerprintAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("fingerprint", Config.FINGERPRINT_THRESHOLD)
    
    def preprocess_data(self, fingerprint_bytes: bytes) -> Any:
        # TODO: Implement fingerprint preprocessing
        raise NotImplementedError("Fingerprint authentication not yet implemented")
    
    def extract_features(self, fingerprint_data: Any) -> np.ndarray:
        # TODO: Implement fingerprint feature extraction
        raise NotImplementedError("Fingerprint authentication not yet implemented")

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
                # Test if the method is implemented
                if method == "voice":
                    available.append(method)
                elif method == "face":
                    available.append(method)  # Add when implemented
                elif method == "fingerprint":
                    available.append(method)
                # Add other methods when implemented
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
    return send_from_directory(Config.STATIC_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """Serve static files"""
    return send_from_directory(Config.STATIC_DIR, path)

@app.route('/api/methods', methods=['GET'])
def get_available_methods():
    """Get available authentication methods"""
    return jsonify({
        "available_methods": biometric_manager.get_available_methods(),
        "all_methods": list(biometric_manager.authenticators.keys())
    })

@app.route('/api/register', methods=['POST'])
def register():
    """Register user with biometric data"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            # Handle file uploads for fingerprint
            if 'fingerprint' in request.files:
                fingerprint_file = request.files['fingerprint']
                if fingerprint_file.filename:
                    # Convert file to base64
                    file_data = fingerprint_file.read()
                    data['data_b64'] = base64.b64encode(file_data).decode('utf-8')
                    data['auth_method'] = 'fingerprint'
        
        # Map frontend field names to backend names
        user_id = data.get('username') or data.get('user_id')
        auth_type = data.get('auth_method') or data.get('auth_type')
        
        # Get the appropriate data field
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
        elif auth_type == 'face':
            data_b64 = data.get('face_data') or data.get('data_b64')
        elif auth_type == 'fingerprint':
            data_b64 = data.get('data_b64')  # Already set above from file
        else:
            data_b64 = data.get('data_b64')
        
        # Validate required fields
        if not all([user_id, auth_type, data_b64]):
            missing = []
            if not user_id: missing.append('username/user_id')
            if not auth_type: missing.append('auth_method/auth_type') 
            if not data_b64: missing.append('biometric data')
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
        
        # Decode data
        try:
            raw_data = decode_base64(data_b64)
        except ValueError as e:
            return jsonify({"success": False, "message": str(e)}), 400
        
        # Register user
        result = authenticator.register_user(user_id, raw_data)
        
        status_code = 200 if result["success"] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate user with biometric data"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            # Handle file uploads for fingerprint
            if 'fingerprint' in request.files:
                fingerprint_file = request.files['fingerprint']
                if fingerprint_file.filename:
                    # Convert file to base64
                    file_data = fingerprint_file.read()
                    data['data_b64'] = base64.b64encode(file_data).decode('utf-8')
                    data['auth_method'] = 'fingerprint'
        
        # Map frontend field names to backend names
        user_id = data.get('username') or data.get('user_id')
        auth_type = data.get('auth_method') or data.get('auth_type')
        
        print(f"Login attempt: {user_id} using {auth_type}")
        
        # Get the appropriate data field
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
        elif auth_type == 'face':
            data_b64 = data.get('face_data') or data.get('data_b64')
        elif auth_type == 'fingerprint':
            data_b64 = data.get('data_b64')  # Already set above from file
        else:
            data_b64 = data.get('data_b64')
        
        # Validate required fields
        if not all([user_id, auth_type, data_b64]):
            missing = []
            if not user_id: missing.append('username/user_id')
            if not auth_type: missing.append('auth_method/auth_type')
            if not data_b64: missing.append('biometric data')
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
        
        # Decode data
        try:
            raw_data = decode_base64(data_b64)
        except ValueError as e:
            return jsonify({"success": False, "message": str(e)}), 400
        
        # Authenticate user
        result = authenticator.authenticate_user(user_id, raw_data)
        
        status_code = 200 if result["success"] else 401
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/user/<user_id>/methods', methods=['GET'])
def get_user_registered_methods(user_id):
    """Get authentication methods registered for a specific user"""
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
    deleted_methods = []
    errors = []
    
    for method_name, authenticator in biometric_manager.authenticators.items():
        try:
            feature_path = os.path.join(authenticator.data_dir, f"{user_id}.npy")
            if os.path.exists(feature_path):
                os.remove(feature_path)
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
    return jsonify({"success": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "message": "Internal server error"}), 500

if __name__ == '__main__':
    print("Available authentication methods:", biometric_manager.get_available_methods())
    app.run(debug=True, host='0.0.0.0', port=5000 )