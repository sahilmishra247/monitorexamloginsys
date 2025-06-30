# authentication.py
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
import cv2
import mysql.connector
from mysql.connector import Error

try:
    import cv2
    # Test if face recognition module is available
    test_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("OpenCV face recognition module loaded successfully")
except AttributeError:
    print("ERROR: OpenCV face recognition module not available")
    print("Install opencv-contrib-python: pip install opencv-contrib-python")
    exit(1)

app = Flask(__name__)
CORS(app)

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    #'ssl_disabled': True,  # Temporarily disable SSL for testing
    # 'ssl_ca': '/etc/ssl/cert.pem',  # Comment out for now
}

class Config:
    SAMPLE_RATE = 16000
    VOICE_THRESHOLD = 0.75
    EMBEDDING_DIR = "monitorexamloginsys/embeddings"
    STATIC_DIR = "static"
    
    # Thresholds for biometric methods
    FACE_THRESHOLD = 0.70  # Adjusted for OpenCV's LBPH recognizer
    FINGERPRINT_THRESHOLD = 200  # Adjusted to match app.py's threshold of 200 matches

def db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Database connection error: {e}")
        return None

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

    def delete_features(self, user_id: str) -> bool:
        """Delete biometric data (face or voice) for the user from the database, and remove user if all are null"""
        conn = None
        cursor = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Could not connect to database")

            cursor = conn.cursor()
            column_name = {
                'face': 'face_data',
                'voice': 'voice_data'
            }.get(self.auth_type)

            if not column_name:
                raise ValueError(f"Unsupported auth type for deletion: {self.auth_type}")

            update_query = f"""
                UPDATE users_biometrics
                SET {column_name} = NULL
                WHERE username = %s
            """
            cursor.execute(update_query, (user_id,))
            conn.commit()

            check_query = """
                SELECT voice_data, face_data, fingerprint_data
                FROM users_biometrics
                WHERE username = %s
            """
            cursor.execute(check_query, (user_id,))
            result = cursor.fetchone()

            if result and all(field is None for field in result):
                print(f"All biometrics null for '{user_id}'. Deleting user entry.")
                delete_query = "DELETE FROM users_biometrics WHERE username = %s"
                cursor.execute(delete_query, (user_id,))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting {self.auth_type} data for user '{user_id}': {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()
        
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        a = np.array(features1)
        b = np.array(features2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def save_features(self, user_id: str, features: np.ndarray) -> bool:
        """Save user features as a blob in a database"""
        cursor = None
        conn = None
        try:
            conn=db_connection()
            if conn is None:
                raise ValueError("Database connection failed")
            
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM users_biometrics WHERE username = %s", (user_id,))
            if cursor.fetchone():
                print("User already registered.")
                return False

            query = f"""
                INSERT INTO users_biometrics (username, {self.auth_type}_data)
                VALUES (%s, %s)
            """
            # Convert NumPy array to bytes
            embedding_blob = features.tobytes()
            cursor.execute(query, (user_id, embedding_blob))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving features: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()
    
    def load_features(self, user_id: str) -> Optional[np.ndarray]:
        """Load user features from file"""
        cursor = None
        conn = None
        try:
            conn = db_connection()
            if conn is None:
                raise ValueError("Database connection failed")
            
            cursor = conn.cursor()
            query = f"SELECT {self.auth_type}_data FROM users_biometrics WHERE username = %s"
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

            if row is None or row[0] is None:
                print(f"No {self.auth_type} features found for user '{user_id}'")
                return None
            
            (row_data,) = row

            if not isinstance(row_data, (bytes, bytearray)):
                raise ValueError("Expected BLOB data in bytes format")
            
            embedding = np.frombuffer(row_data, dtype=np.float32)
            return embedding
        
        except Exception as e:
            print(f"Error loading {self.auth_type} features: {e}")
            return None
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()
    
    def register_user(self, user_id: str, data: Any) -> Dict[str, Any]:
        """Register a new user"""
        print(f"Registering user '{user_id}' for {self.auth_type}")
        
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
            test_features = self.extract_features(processed_data)
            
            # Calculate similarity based on auth type
            if self.auth_type == "fingerprint":
                similarity = self.calculate_similarity(test_features, stored_features)
            else:
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

# Fingerprint Authentication Implementation
class FingerprintAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("fingerprint", Config.FINGERPRINT_THRESHOLD)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.allowed_extensions = {'.jpg', '.jpeg', '.png'}
    
    def preprocess_data(self, data):
        """Preprocess fingerprint image data"""
        try:
            if hasattr(data, 'read'):  # File-like object
                image_bytes = data.read()
            else:
                image_bytes = data
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError("Could not decode fingerprint image")
            
            return image
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
                'descriptors': descriptors,
                'image': image_data
            }
        except Exception as e:
            print(f"Error extracting fingerprint features: {str(e)}")
            raise ValueError(f"Error extracting fingerprint features: {str(e)}")
    
    def save_features(self, user_id: str, features: np.ndarray) -> bool:
        """Save voice features to database"""
        cursor = None
        conn = None
        try:
            conn = db_connection()
            if conn is None:
                raise ValueError("Database connection failed")
            
            cursor = conn.cursor()

            # Check if user exists and update or insert
            cursor.execute("SELECT id FROM users_biometrics WHERE username = %s", (user_id,))
            if cursor.fetchone():
                query = """
                    UPDATE users_biometrics 
                    SET voice_data = %s
                    WHERE username = %s
                """
                cursor.execute(query, (features.tobytes(), user_id))
            else:
                query = """
                    INSERT INTO users_biometrics (username, voice_data)
                    VALUES (%s, %s)
                """
                cursor.execute(query, (user_id, features.tobytes()))

            conn.commit()
            print("Successfully saved voice data")
            return True
        except Exception as e:
            print(f"Error saving voice features: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    def load_features(self, user_id: str) -> Optional[dict]:
        """Load fingerprint features from database"""
        cursor = None
        conn = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Could not establish DB connection")

            cursor = conn.cursor()
            query = "SELECT fingerprint_data FROM users_biometrics WHERE username = %s"
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

            if row is None or row[0] is None:
                print(f"No fingerprint features found for user '{user_id}'")
                return None
            
            (row_data,) = row
            image_data = np.frombuffer(row_data, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print("Failed to decode fingerprint image from DB")
                return None

            # Extract features from the loaded image
            features = self.extract_features(image)
            return features

        except Exception as e:
            print(f"Error loading fingerprint from database: {e}")
            return None
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

# Face Authentication Implementation using OpenCV
class FaceAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("face", Config.FACE_THRESHOLD)
        # Initialize face detector and recognizer
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    def preprocess_data(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess face image data"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode face image")
            
            # Convert to grayscale (face detection works better on grayscale)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30))
            
            if len(faces) == 0:
                raise ValueError("No faces detected in the image")
            
            # Return the first face found
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to a standard size for consistency
            face_roi = cv2.resize(face_roi, (100, 100))
            
            return face_roi
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise ValueError(f"Error preprocessing face image: {str(e)}")
    
    def extract_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face features - return the preprocessed image itself"""
        try:
            return face_image
        except Exception as e:
            print(f"Error extracting face features: {str(e)}")
            raise ValueError(f"Error extracting face features: {str(e)}")
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two face images using OpenCV's LBPH recognizer"""
        try:
            # Train a temporary recognizer with one sample
            self.face_recognizer.train([features1], np.array([0]))
            
            # Predict the label and confidence for the test image
            label, confidence = self.face_recognizer.predict(features2)
            
            # Convert confidence to similarity score
            # LBPH returns 0 for perfect match, higher values for worse matches
            # We'll invert this to get a similarity score between 0 and 1
            similarity = max(0, 1 - (confidence / 100))
            return similarity
        except Exception as e:
            print(f"Error calculating face similarity: {e}")
            return 0.0
    
    def save_features(self, user_id: str, features: np.ndarray) -> bool:
        """Save face features to database"""
        cursor = None
        conn = None
        try:
            # Encode the face image as PNG
            success, buffer = cv2.imencode('.png', features)
            if not success:
                raise ValueError("Failed to encode face image")

            image_bytes = buffer.tobytes()

            conn = db_connection()
            if not conn:
                raise ValueError("Could not establish DB connection")

            cursor = conn.cursor()

            # Check if user exists and update or insert
            cursor.execute("SELECT id FROM users_biometrics WHERE username = %s", (user_id,))
            if cursor.fetchone():
                query = """
                    UPDATE users_biometrics 
                    SET face_data = %s
                    WHERE username = %s
                """
                cursor.execute(query, (image_bytes, user_id))
            else:
                query = """
                    INSERT INTO users_biometrics (username, face_data)
                    VALUES (%s, %s)
                """
                cursor.execute(query, (user_id, image_bytes))

            conn.commit()
            print("Successfully saved face data")
            return True

        except Exception as e:
            print(f"Error saving face image to database: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    def load_features(self, user_id: str) -> Optional[np.ndarray]:
        """Load the face image from DB"""
        cursor = None
        conn = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Could not establish DB connection")

            cursor = conn.cursor()
            query = "SELECT face_data FROM users_biometrics WHERE username = %s"
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

            if row is None or row[0] is None:
                print(f"No {self.auth_type} features found for user '{user_id}'")
                return None
            
            (row_data,) = row
            image_data = np.frombuffer(row_data, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print("Failed to decode face image from DB")
                return None

            return image

        except Exception as e:
            print(f"Error loading face image from database: {e}")
            return None

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

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
                    available.append(method)
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

@app.route('/api/db-test')
def db_test():
    try:
        conn = db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            return jsonify([table[0] for table in tables])
        return "Connection failed", 500
    except Exception as e:
        return str(e), 500

@app.route('/api/user/<user_id>/delete', methods=['DELETE'])
def delete_user_data(user_id):
    """Delete all biometric data for a user"""
    print(f"Received /api/user/{user_id}/delete request")
    deleted_methods = []
    errors = []

    for method_name, authenticator in biometric_manager.authenticators.items():
        try:
            success = authenticator.delete_features(user_id)
            if success:
                deleted_methods.append(method_name)
            else:
                errors.append(f"Failed to delete {method_name} data for user '{user_id}'")
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

def create_tables():
    """Create the users_biometrics table if it doesn't exist"""
    conn = None
    cursor = None
    try:
        conn = db_connection()
        if not conn:
            print("Could not connect to database")
            return False
        
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users_biometrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            voice_data LONGBLOB NULL,
            face_data LONGBLOB NULL,
            fingerprint_data LONGBLOB NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        print("Database table created/verified successfully")
        return True
        
    except Exception as e:
        print(f"Error creating database table: {e}")
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()

if __name__ == '__main__':
    print("Starting Flask server...")
    
    # Create database tables
    if not create_tables():
        print("Failed to create/verify database tables")
        exit(1)
    
    print("Available authentication methods:", biometric_manager.get_available_methods())
    app.run(debug=True, host='0.0.0.0', port=5000)