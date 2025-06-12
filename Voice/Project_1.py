'''from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import os
import librosa
import random
from typing import List, Optional
from pydantic import BaseModel
from resemblyzer import VoiceEncoder
import io

# --- FastAPI app setup ---
app = FastAPI()

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Parameters ---
SAMPLE_RATE = 16000
THRESHOLD = 0.85
EMBEDDING_DIR = "IdeaProjects/Innokreat/embeddings"

# --- Directory Setup ---
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# --- Helper Functions ---
def random_line_from_file(filepath):
    total_words = [10, 11, 12]
    sentence = ""
    for n in total_words:    
        target_index = random.randint(0, n - 1)
        filepath = os.path.join("IdeaProjects","Innokreat","words", f"word{total_words.index(n)}.txt")
        with open(filepath, 'r') as f:
            current_index = 0
            for line in f:
                words = line.strip().split()
                if current_index + len(words) > target_index:
                    sentence += words[target_index - current_index]
                    break
                current_index += len(words)
    return sentence

def decode_base64(b64_string: str) -> bytes:
    try:
        return base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def preprocess_audio_file(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if np.max(np.abs(y_trimmed)) > 0:
        y_normalized = y_trimmed / np.max(np.abs(y_trimmed))
    else:
        y_normalized = y_trimmed
    return y_normalized

def extract_mfcc(y: np.ndarray) -> np.ndarray:
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(y)
    return embedding

# --- FastAPI Routes ---
class AudioBase64Payload(BaseModel):
    user_id: str
    audio_b64: Optional[str] = None
    action: Optional[str] = None  # "register" or "login"

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/register")
async def register(payload: AudioBase64Payload):
    user_id = payload.user_id

    if not user_id or not payload.audio_b64:
        raise HTTPException(status_code=400, detail="Missing user ID or audio data")
    
    audio_bytes = decode_base64(payload.audio_b64)
    embed_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")

    if os.path.exists(embed_path):
        return JSONResponse(content={"message": f"User '{user_id}' already registered."})

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    y = preprocess_audio_file(y)
    embedding = extract_mfcc(y)
    np.save(embed_path, embedding)
    return JSONResponse(content={"message": f"Registered '{user_id}' successfully."})

@app.post("/login")
async def login(payload: AudioBase64Payload):
    user_id = payload.user_id
    embed_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")

    if not os.path.exists(embed_path):
        raise HTTPException(status_code=404, detail="No registered voiceprint found for this user.")

    stored_embedding = np.load(embed_path)
    audio_bytes = decode_base64(payload.audio_b64)
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    y = preprocess_audio_file(y)
    test_embedding = extract_mfcc(y)
    similarity = cosine_similarity(test_embedding, stored_embedding)

    if similarity >= THRESHOLD:
        return JSONResponse(content={"message": f"Login successful for '{user_id}' (Similarity: {similarity:.3f})"})
    else:
        return JSONResponse(content={"message": f"Authentication failed for '{user_id}' (Similarity: {similarity:.3f})"})
'''

# voicelogin.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import os
import librosa
from resemblyzer import VoiceEncoder
import io

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
SAMPLE_RATE = 16000
THRESHOLD = 0.85
EMBEDDING_DIR = "IdeaProjects/Innokreat/embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Static files (frontend)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Helper functions
def decode_base64(b64_string: str) -> bytes:
    try:
        return base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

def cosine_similarity(v1, v2) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def preprocess_audio(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed / np.max(np.abs(y_trimmed)) if np.max(np.abs(y_trimmed)) > 0 else y_trimmed

def extract_mfcc(y: np.ndarray) -> np.ndarray:
    encoder = VoiceEncoder()
    return encoder.embed_utterance(y)

# Request model
class AudioBase64Payload(BaseModel):
    user_id: str
    audio_b64: str

# API endpoints
@app.post("/api/register")
async def register(payload: AudioBase64Payload):
    user_id = payload.user_id
    embed_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")

    if os.path.exists(embed_path):
        return JSONResponse(content={"message": f"User '{user_id}' already registered."})

    audio_bytes = decode_base64(payload.audio_b64)
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    y = preprocess_audio(y)
    embedding = extract_mfcc(y)
    np.save(embed_path, embedding)

    return JSONResponse(content={"message": f"Registered '{user_id}' successfully."})

@app.post("/api/login")
async def login(payload:AudioBase64Payload):
    print(f"Login called with: {payload.user_id}, audio length: {len(payload.audio_b64 or '')}")
    user_id = payload.user_id
    embed_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")

    if not os.path.exists(embed_path):
        raise HTTPException(status_code=404, detail="No registered voiceprint found for this user.")

    stored_embedding = np.load(embed_path)
    audio_bytes = decode_base64(payload.audio_b64)
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    y = preprocess_audio(y)
    test_embedding = extract_mfcc(y)
    similarity = cosine_similarity(test_embedding, stored_embedding)

    if similarity >= THRESHOLD:
        return JSONResponse(content={"message": f"Login successful for '{user_id}' (Similarity: {similarity:.3f})"})
    else:
        return JSONResponse(content={"message": f"Authentication failed for '{user_id}' (Similarity: {similarity:.3f})"})
