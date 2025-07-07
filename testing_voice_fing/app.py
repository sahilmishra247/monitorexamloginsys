# authentication.py
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import base64
import numpy as np
import os
import librosa
from resemblyzer import VoiceEncoder
import io
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import secrets
from flask_session import Session
import mysql.connector
from mysql.connector import Error

# Core WebAuthn library functions
from webauthn import (
    generate_registration_options,
    verify_registration_response,
    generate_authentication_options,
    verify_authentication_response,
)

# CRUCIAL FIX: Import AttestedCredential and AuthenticationCredential from webauthn.types
from webauthn.types import (
    AttestedCredential,
    AuthenticationCredential,
)

# Other data structures and enums from webauthn.helpers.structs
from webauthn.helpers.structs import (
    PublicKeyCredentialRpEntity,
    PublicKeyCredentialUserEntity,
    AuthenticatorSelectionCriteria,
    AttestationConveyancePreference,
    UserVerificationRequirement,
    ResidentKeyRequirement,
    AuthenticatorAttachment,
    AuthenticatorAttestationResponse,  # Response structs for attestation
    AuthenticatorAssertionResponse,  # Response structs for assertion
    PublicKeyCredentialDescriptor,
    PublicKeyCredentialType,
    COSEAlgorithmIdentifier,
)

# Helper functions for base64 encoding/decoding
from webauthn.helpers import (
    bytes_to_base64url,
    base64url_to_bytes,
)

app = Flask(__name__)

# --- Flask-Session Configuration ---
# This is crucial for storing WebAuthn challenges between requests
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
# Ensure the session directory exists if using filesystem session type
if app.config['SESSION_TYPE'] == 'filesystem':
    app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)
# --- End Flask-Session Configuration ---

CORS(app)

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'authuser',
    'password': 'securepass123',
    'database': 'biometric_auth'
}


class Config:
    SAMPLE_RATE = 16000
    VOICE_THRESHOLD = 0.75
    EMBEDDING_DIR = "monitorexamloginsys/embeddings"
    STATIC_DIR = "static"


def db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Database connection error: {e}")
        return None


# Ensure directories exist
os.makedirs(Config.EMBEDDING_DIR, exist_ok=True)
os.makedirs(Config.STATIC_DIR, exist_ok=True)


# --- WebAuthn Database Helper Functions ---
def save_webauthn_credential_to_db(username: str, credential_data: Dict[str, Any]) -> bool:
    """
    Saves a WebAuthn credential for a user in the database.
    If the user exists, it appends to their existing credentials.
    Otherwise, it creates a new user entry with the credential.
    """
    conn = None
    cursor = None
    try:
        conn = db_connection()
        if not conn:
            raise ValueError("Could not connect to database")

        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT webauthn_credentials FROM users_biometrics WHERE username = %s", (username,))
        row = cursor.fetchone()

        current_credentials = []
        if row and row[0]:
            try:
                current_credentials = json.loads(row[0])
            except json.JSONDecodeError:
                print(f"Warning: Existing webauthn_credentials for {username} is not valid JSON. Overwriting.")
                current_credentials = []

        # Add the new credential
        current_credentials.append(credential_data)
        updated_credentials_json = json.dumps(current_credentials)

        if row:
            # Update existing user
            query = """
                UPDATE users_biometrics
                SET webauthn_credentials = %s
                WHERE username = %s
            """
            cursor.execute(query, (updated_credentials_json, username))
        else:
            # Insert new user with webauthn credential
            query = """
                INSERT INTO users_biometrics (username, webauthn_credentials)
                VALUES (%s, %s)
            """
            cursor.execute(query, (username, updated_credentials_json))

        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving WebAuthn credential for user '{username}': {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def load_webauthn_credentials_from_db(username: str) -> List[Dict[str, Any]]:
    """Loads all WebAuthn credentials for a given user from the database."""
    conn = None
    cursor = None
    try:
        conn = db_connection()
        if not conn:
            raise ValueError("Could not connect to database")

        cursor = conn.cursor()
        query = "SELECT webauthn_credentials FROM users_biometrics WHERE username = %s"
        cursor.execute(query, (username,))
        row = cursor.fetchone()

        if row and row[0]:
            return json.loads(row[0])
        return []
    except Exception as e:
        print(f"Error loading WebAuthn credentials for user '{username}': {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def delete_webauthn_credentials_from_db(username: str) -> bool:
    """Deletes all WebAuthn credentials for a user from the database."""
    conn = None
    cursor = None
    try:
        conn = db_connection()
        if not conn:
            raise ValueError("Could not connect to database")

        cursor = conn.cursor()
        update_query = """
            UPDATE users_biometrics
            SET webauthn_credentials = NULL
            WHERE username = %s
        """
        cursor.execute(update_query, (username,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting WebAuthn credentials for user '{username}': {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


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
        """
        Delete biometric data for the user from the database.
        If all biometric data (voice, webauthn) are null,
        the user entry is removed from the database.
        """
        conn = None
        cursor = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Could not connect to database")

            cursor = conn.cursor()
            column_name = {
                'voice': 'voice_data',
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

            # Check if all biometrics are now NULL
            check_query = """
                SELECT voice_data, webauthn_credentials
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
            conn = db_connection()
            if conn is None:
                raise ValueError("Database connection failed")

            cursor = conn.cursor()

            # Check if user already exists
            cursor.execute("SELECT id FROM users_biometrics WHERE username = %s", (user_id,))
            user_exists = cursor.fetchone()

            if user_exists:
                # If user exists, update the specific biometric data
                query = f"""
                    UPDATE users_biometrics
                    SET {self.auth_type}_data = %s
                    WHERE username = %s
                """
                embedding_blob = features.tobytes() if isinstance(features, np.ndarray) else features
                cursor.execute(query, (embedding_blob, user_id))
                conn.commit()
                print(f"Updated {self.auth_type} data for existing user '{user_id}'.")
                return True
            else:
                # If user does not exist, insert new record
                query = f"""
                    INSERT INTO users_biometrics (username, {self.auth_type}_data)
                    VALUES (%s, %s)
                """
                embedding_blob = features.tobytes() if isinstance(features, np.ndarray) else features
                cursor.execute(query, (user_id, embedding_blob))
                conn.commit()
                print(f"Registered new user '{user_id}' with {self.auth_type} data.")
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


# Biometric Manager
class BiometricManager:
    def __init__(self):
        self.authenticators = {
            "voice": VoiceAuthenticator(),
        }

    def get_authenticator(self, auth_type: str) -> Optional[BiometricAuthenticator]:
        return self.authenticators.get(auth_type)

    def get_available_methods(self) -> list:
        """Return list of available authentication methods"""
        available = []
        for method, authenticator in self.authenticators.items():
            try:
                available.append(method)
            except NotImplementedError:
                pass
        # Add webauthn explicitly as it's handled separately
        available.append("webauthn")
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
        "all_methods": list(biometric_manager.authenticators.keys()) + ["webauthn"]
    })


@app.route('/api/register', methods=['POST'])
def register():
    """Register user with biometric data (excluding WebAuthn, which has its own flow)"""
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

        # WebAuthn has its own registration flow, so it's not handled here
        if auth_type == 'webauthn':
            return jsonify({
                "success": False,
                "message": "WebAuthn registration requires a separate endpoint."
            }), 400

        # Get the appropriate data
        raw_data = None
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing voice biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        else:
            return jsonify({
                "success": False,
                "message": "No biometric data provided or unsupported authentication type"
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
    """Authenticate user with biometric data (excluding WebAuthn, which has its own flow)"""
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

        # WebAuthn has its own login flow, so it's not handled here
        if auth_type == 'webauthn':
            return jsonify({
                "success": False,
                "message": "WebAuthn login requires a separate endpoint."
            }), 400

        print(f"Login attempt: {user_id} using {auth_type}")

        # Get the appropriate data
        raw_data = None
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing voice biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        else:
            return jsonify({
                "success": False,
                "message": "No biometric data provided or unsupported authentication type"
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

    # Check for voice
    voice_authenticator = biometric_manager.get_authenticator("voice")
    if voice_authenticator and voice_authenticator.load_features(user_id) is not None:
        registered_methods.append("voice")

    # Check for WebAuthn
    webauthn_creds = load_webauthn_credentials_from_db(user_id)
    if webauthn_creds:
        registered_methods.append("webauthn")

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

    # Delete voice data
    voice_authenticator = biometric_manager.get_authenticator("voice")
    if voice_authenticator:
        try:
            success = voice_authenticator.delete_features(user_id)
            if success:
                deleted_methods.append("voice")
            else:
                errors.append(f"Failed to delete voice data for user '{user_id}'")
        except Exception as e:
            errors.append(f"Error deleting voice data: {str(e)}")

    # Delete WebAuthn data
    try:
        success = delete_webauthn_credentials_from_db(user_id)
        if success:
            deleted_methods.append("webauthn")
        else:
            errors.append(f"Failed to delete webauthn data for user '{user_id}'")
    except Exception as e:
        errors.append(f"Error deleting webauthn data: {str(e)}")

    # After attempting to delete all types, check if the user entry should be removed
    conn = None
    cursor = None
    try:
        conn = db_connection()
        if conn:
            cursor = conn.cursor()
            check_query = """
                SELECT voice_data, webauthn_credentials
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
    except Exception as e:
        print(f"Error during final user entry cleanup for '{user_id}': {e}")
        errors.append(f"Error during final user entry cleanup: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

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


# --- WebAuthn Specific Routes ---
@app.route('/register-webauthn-challenge', methods=['POST'])
def register_webauthn_challenge():
    """Generates WebAuthn registration options (challenge) for the frontend."""
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    # In a real app, you'd check if the username exists in your primary user table
    # For this example, we check if they have any existing webauthn credentials
    existing_credentials = load_webauthn_credentials_from_db(username)
    if existing_credentials:
        # If user already has passkeys, you might want to allow adding more,
        # or prevent re-registration of the same username if it implies a new user.
        # For simplicity, we'll allow adding more credentials to an existing user.
        pass # Allow adding more credentials

    # Generate a unique user ID for WebAuthn if it's a new user, or retrieve existing
    # For this example, we'll generate a new user_id if one doesn't exist for webauthn.
    # In a real app, this user_id should be consistent across all credentials for a user.
    user_id_bytes = None
    if existing_credentials:
        # Try to use the user_id from an existing credential, if available
        # This is a simplification; ideally, user_id should be consistent across all credentials for a user
        first_cred = existing_credentials[0]
        if 'user_id' in first_cred:
            user_id_bytes = base64url_to_bytes(first_cred['user_id'])
        else:
            # If old credentials don't have user_id, generate a new one for this registration
            user_id_bytes = secrets.token_bytes(16)
    else:
        user_id_bytes = secrets.token_bytes(16)


    user_entity = PublicKeyCredentialUserEntity(
        id=user_id_bytes,
        name=username,
        display_name=username
    )

    registration_options = generate_registration_options(
        rp_id=request.host.split(':')[0],
        rp_name="My Secure WebAuthn App",
        user=user_entity,
        challenge=secrets.token_bytes(32),
        authenticator_selection=AuthenticatorSelectionCriteria(
            authenticator_attachment=AuthenticatorAttachment.PLATFORM,
            user_verification=UserVerificationRequirement.REQUIRED,
            resident_key=ResidentKeyRequirement.PREFERRED,
        ),
        timeout=60000,
        attestation=AttestationConveyancePreference.DIRECT,
        pub_key_cred_params=[
            COSEAlgorithmIdentifier.ES256,
            COSEAlgorithmIdentifier.RS256,
        ]
    )

    # Store challenge and user info in session for verification step
    session['challenge'] = bytes_to_base64url(registration_options.challenge)
    session['registering_user_id'] = bytes_to_base64url(user_id_bytes)
    session['registering_username'] = username

    options_dict = registration_options.dict()

    return jsonify({'publicKey': options_dict})


@app.route('/register-webauthn-verify', methods=['POST'])
def register_webauthn_verify():
    """Verifies the WebAuthn registration response from the frontend."""
    data = request.json

    challenge = session.pop('challenge', None)
    registering_user_id = session.pop('registering_user_id', None)
    registering_username = session.pop('registering_username', None)

    if not challenge or not registering_user_id or not registering_username:
        print("Registration verification failed: Session data missing.")
        return jsonify({'error': 'Session data missing. Please restart registration.'}), 400

    try:
        attested_credential = AttestedCredential(
            id=data['id'],
            raw_id=base64url_to_bytes(data['rawId']),
            type=data['type'],
            response=AuthenticatorAttestationResponse(
                client_data_json=base64url_to_bytes(data['response']['clientDataJSON']),
                attestation_object=base64url_to_bytes(data['response']['attestationObject'])
            ),
            authenticator_attachment=data.get('authenticatorAttachment'),
            client_extension_results=data.get('clientExtensionResults', {}),
        )

        verified_credential = verify_registration_response(
            credential=attested_credential,
            expected_challenge=base64url_to_bytes(challenge),
            expected_origin=request.url_root.rstrip('/'),
            expected_rp_id=request.host.split(':')[0],
            require_user_verification=True
        )

        # Prepare credential data for storage
        credential_to_store = {
            'credential_id': bytes_to_base64url(verified_credential.credential_id),
            'public_key': bytes_to_base64url(verified_credential.public_key),
            'sign_count': verified_credential.sign_count,
            'user_id': bytes_to_base64url(verified_credential.user_id), # Store user_id associated with this credential
            'transports': verified_credential.transports # Store transports if available
        }

        if save_webauthn_credential_to_db(registering_username, credential_to_store):
            print(f"User '{registering_username}' successfully registered credential: {verified_credential.credential_id.hex()}")
            return jsonify({'success': True, 'message': 'Credential registered successfully!'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save credential to database.'}), 500

    except Exception as e:
        print(f"Registration verification failed: {e}")
        return jsonify({'success': False, 'error': f'Registration failed: {e}'}), 400


@app.route('/login-webauthn-challenge', methods=['POST'])
def login_webauthn_challenge():
    """Generates WebAuthn authentication options (challenge) for the frontend."""
    challenge = secrets.token_bytes(32)
    
    data = request.json
    username = data.get('username') # Optional: if user provides username to narrow down credentials

    allow_credentials = []
    if username:
        user_credentials_data = load_webauthn_credentials_from_db(username)
        for cred_data in user_credentials_data:
            allow_credentials.append(
                PublicKeyCredentialDescriptor(
                    id=base64url_to_bytes(cred_data['credential_id']),
                    type=PublicKeyCredentialType.PUBLIC_KEY,
                    # transports=cred_data.get('transports') # Uncomment if you stored transports
                )
            )
    # If no username provided, allow discovery of resident credentials (passkeys)
    # The authenticator will prompt the user to select a passkey.

    authentication_options = generate_authentication_options(
        rp_id=request.host.split(':')[0],
        challenge=challenge,
        allow_credentials=allow_credentials, # Empty list enables discoverable credentials
        user_verification=UserVerificationRequirement.REQUIRED,
        timeout=60000
    )

    session['challenge'] = bytes_to_base64url(authentication_options.challenge)

    options_dict = authentication_options.dict()

    # Convert bytes to base64url for JSON serialization for allowCredentials
    if options_dict.get('allowCredentials'):
        for cred in options_dict['allowCredentials']:
            cred['id'] = bytes_to_base64url(cred['id'])

    return jsonify({'publicKey': options_dict})


@app.route('/login-webauthn-verify', methods=['POST'])
def login_webauthn_verify():
    """Verifies the WebAuthn authentication response from the frontend."""
    data = request.json

    challenge = session.pop('challenge', None)

    if not challenge:
        print("Authentication verification failed: Session data missing.")
        return jsonify({'error': 'Session data missing. Please restart login.'}), 400

    try:
        authentication_credential = AuthenticationCredential(
            id=data['id'],
            raw_id=base64url_to_bytes(data['rawId']),
            type=data['type'],
            response=AuthenticatorAssertionResponse(
                client_data_json=base64url_to_bytes(data['response']['clientDataJSON']),
                authenticator_data=base64url_to_bytes(data['response']['authenticatorData']),
                signature=base64url_to_bytes(data['response']['signature']),
                user_handle=base64url_to_bytes(data['response'].get('userHandle', '')) if data['response'].get('userHandle') else None
            ),
            client_extension_results=data.get('clientExtensionResults', {})
        )

        found_credential_data = None
        current_username = None

        # Find the credential in the database based on raw_id (credential_id)
        # Iterate through all users to find the matching credential
        conn = db_connection()
        if not conn:
            raise ValueError("Could not connect to database")
        cursor = conn.cursor()
        cursor.execute("SELECT username, webauthn_credentials FROM users_biometrics WHERE webauthn_credentials IS NOT NULL")
        all_users_creds = cursor.fetchall()
        cursor.close()
        conn.close()

        for username_db, creds_json in all_users_creds:
            try:
                creds_list = json.loads(creds_json)
                for cred_item in creds_list:
                    if base64url_to_bytes(cred_item['credential_id']) == authentication_credential.raw_id:
                        found_credential_data = cred_item
                        current_username = username_db
                        break
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON for webauthn_credentials for user {username_db}")
            if found_credential_data:
                break

        if not found_credential_data:
            print(f"Login verification failed: Credential {data['id']} not found in database.")
            return jsonify({'success': False, 'error': 'Credential not found or invalid.'}), 401

        verified_credential = verify_authentication_response(
            credential=authentication_credential,
            expected_challenge=base64url_to_bytes(challenge),
            expected_origin=request.url_root.rstrip('/'),
            expected_rp_id=request.host.split(':')[0],
            credential_public_key=base64url_to_bytes(found_credential_data['public_key']),
            credential_sign_count=found_credential_data['sign_count'],
            require_user_verification=True
        )

        # Update sign_count in the database
        # Find the specific credential in the list and update its sign_count
        user_credentials = load_webauthn_credentials_from_db(current_username)
        for i, cred_item in enumerate(user_credentials):
            if base64url_to_bytes(cred_item['credential_id']) == authentication_credential.raw_id:
                user_credentials[i]['sign_count'] = verified_credential.sign_count
                break
        
        # Save the updated list back to the database
        if not save_webauthn_credential_to_db(current_username, user_credentials):
            print(f"Warning: Failed to update sign_count for user {current_username}")

        print(f"User '{current_username}' successfully logged in with credential: {verified_credential.credential_id.hex()}")
        return jsonify({'success': True, 'message': f'Login successful for {current_username}!'})

    except Exception as e:
        print(f"Authentication verification failed: {e}")
        return jsonify({'success': False, 'error': f'Login failed: {e}'}), 401


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
