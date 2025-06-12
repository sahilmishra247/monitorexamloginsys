from fastapi import FastAPI, HTTPException, UploadFile, Form, Request
import wave
import base64
import pyaudio
import sys
import numpy as np
import os
import librosa
import random
from typing import List, Optional
from pydantic import BaseModel
from resemblyzer import VoiceEncoder
# --- Parameters ---
SAMPLE_RATE = 16000
THRESHOLD = 0.85
EMBEDDING_DIR = "embeddings"
FORMAT = pyaudio.paFloat32
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 5

def record_audio_in_memory():
    print("Please say the phrase 'Hello, this is (your name) and my id is (your user ID)' clearly.")
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_np = np.concatenate(frames)  # Combine into single NumPy array
    return audio_np

def random_line_from_file(filepath):
    total_words = [10,11,12]
    if total_words == 0:
        return None  # no words in file
    sentence = ""
    for n in total_words:    
        # Pick a random word index
        target_index = random.randint(0, n - 1)
        filepath = os.path.join("IdeaProjects","Innokreat","words", f"word{total_words.index(n)}.txt")
        
        # find and return that word
        with open(filepath, 'r') as f:
            current_index = 0
            for line in f:
                words = line.strip().split()
                if current_index + len(words) > target_index:
                    # The target word is in this line
                    sentence += words[target_index - current_index]
                current_index += len(words)
    return sentence


#defining the input data
class AuthPayload(BaseModel):
    user_id: str
    password: Optional[str] = None
    audio_b64: Optional[str] = None
    photo_b64: Optional[str] = None
    fingerprint_b64: Optional[str] = None
    action: str  # "register" or "login"

def decode_base64(b64_string: str) -> bytes:
    try:
        return base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
# the cosine of the angle between two vectors is found using the formula explained in the design document

# --- Directory Setup ---
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# --- Audio Processing ---
def preprocess_audio_file(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if np.max(np.abs(y_trimmed)) > 0:
        y_normalized = y_trimmed / np.max(np.abs(y_trimmed))
    else:
        y_normalized = y_trimmed
    return y_normalized

def extract_mfcc(y):
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(y)
    return embedding

# --- Main Logic ---
def register_user(user_id):
    embed_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")
    if os.path.exists(embed_path):
        reg=input(f"User '{user_id}' already registered, do you want to change the voice data? (yes/no)")
        if reg.lower() == 'no':
            print(f"User '{user_id}' already registered. Use a different user ID.")
            return
        else:
            os.remove(embed_path)
    Current_input = record_audio_in_memory()
    y = preprocess_audio_file(Current_input)
    embedding = extract_mfcc(y)
    save_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")
    np.save(save_path, embedding)
    print(f"Registered '{user_id}' and saved embedding to '{save_path}'.")

def login_user(user_id):
    embed_path = os.path.join(EMBEDDING_DIR, f"{user_id}.npy")
    if not os.path.exists(embed_path):
        print(f"No registered voiceprint found for user '{user_id}'.")
        return
    stored_embedding = np.load(embed_path)
    Current_input = record_audio_in_memory()
    y = preprocess_audio_file(Current_input)
    test_embedding = extract_mfcc(y)
    similarity= cosine_similarity(test_embedding, stored_embedding)
    if similarity >= THRESHOLD:
        print(f"Login successful for '{user_id}' (Similarity: {similarity:.3f})")
    else:
        print(f"Authentication failed for '{user_id}' (Similarity: {similarity:.3f})")

# --- Interactive CLI ---
def main():
    while True:
        action = input("\nType 'register' or 'login' (or 'exit'): ").strip().lower()
        if action == "exit":
            break
        if action not in ("register", "login"):
            print("Invalid command.")
            continue

        user_id = input("Enter user ID: ").strip()
        if not user_id:
            print("User ID cannot be empty.")
            return
        
        print("Assuming you are trying to login through voice authentication.")

        if action == "register":
            register_user(user_id)
        elif action == "login":
            login_user(user_id)
if __name__ == "__main__":
    main()