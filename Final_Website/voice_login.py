import base64
import numpy as np
import os
import librosa
import io
from flask import request
import json

# Try to import resemblyzer, with fallback for demo purposes
try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    print("Warning: resemblyzer not available. Install with: pip install resemblyzer")
    RESEMBLYZER_AVAILABLE = False

# Constants
SAMPLE_RATE = 16000
THRESHOLD = 0.85
EMBEDDING_DIR = "embeddings"

# Create embeddings directory if it doesn't exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)

def decode_base64(b64_string: str) -> bytes:
    """Decode base64 string to bytes"""
    try:
        return base64.b64decode(b64_string)
    except Exception as e:
        raise Exception(f"Invalid base64 string: {str(e)}")

def cosine_similarity(v1, v2) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(v1)
    b = np.array(v2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))

def preprocess_audio(y):
    """Preprocess audio signal"""
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize
    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_normalized = y_trimmed / max_val
    else:
        y_normalized = y_trimmed
    
    return y_normalized

def extract_voice_embedding(y: np.ndarray) -> np.ndarray:
    """Extract voice embedding from audio signal"""
    if not RESEMBLYZER_AVAILABLE:
        # Fallback: use MFCC features for demo
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(y)
    return embedding

def authenticate(flask_request):
    """
    Main authentication function called by the Flask app
    Expected to receive JSON data with:
    - user_id: string
    - action: 'register' or 'login'
    - audio_b64: base64 encoded audio data
    """
    try:
        # Get JSON data from request
        if flask_request.is_json:
            data = flask_request.get_json()
        else:
            # Try to get from form data if not JSON
            data = {
                'user_id': flask_request.form.get('user_id'),
                'action': flask_request.form.get('action'),
                'audio_b64': flask_request.form.get('audio_b64')
            }
        
        user_id = data.get('user_id')
        action = data.get('action')
        audio_b64 = data.get('audio_b64')
        
        if not user_id:
            return {"success": False, "message": "User ID is required."}
        
        if not audio_b64:
            return {"success": False, "message": "Audio data is required."}
        
        if not action:
            return {"success": False, "message": "Action (register/login) is required."}
        
        # Decode audio data
        audio_bytes = decode_base64(audio_b64)
        
        # Load audio using librosa
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
        
        # Preprocess audio
        y = preprocess_audio(y)
        
        # Check if audio is valid
        if len(y) < SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
            return {"success": False, "message": "Audio too short. Please record for at least 1 second."}
        
        # Extract voice embedding
        embedding = extract_voice_embedding(y)
        embed_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")
        
        if action == "register":
            # Check if user already exists
            if os.path.exists(embed_path):
                return {
                    "success": False, 
                    "message": f"User '{user_id}' already registered. Use login instead.",
                    "action": "register"
                }
            
            # Save embedding
            np.save(embed_path, embedding)
            return {
                "success": True, 
                "message": f"User '{user_id}' registered successfully.",
                "action": "register"
            }
            
        elif action == "login":
            # Check if user exists
            if not os.path.exists(embed_path):
                return {
                    "success": False, 
                    "message": f"No registered voiceprint found for user '{user_id}'. Please register first.",
                    "action": "login"
                }
            
            # Load stored embedding
            stored_embedding = np.load(embed_path)
            
            # Calculate similarity
            similarity = cosine_similarity(embedding, stored_embedding)
            
            if similarity >= THRESHOLD:
                return {
                    "success": True, 
                    "message": f"Login successful for '{user_id}' (Similarity: {similarity:.3f})",
                    "action": "login",
                    "similarity": similarity,
                    "redirect": "/success"
                }
            else:
                return {
                    "success": False, 
                    "message": f"Authentication failed for '{user_id}' (Similarity: {similarity:.3f}). Please try again.",
                    "action": "login",
                    "similarity": similarity
                }
        
        else:
            return {"success": False, "message": "Invalid action. Use 'register' or 'login'."}
            
    except Exception as e:
        return {"success": False, "message": f"Authentication error: {str(e)}"}