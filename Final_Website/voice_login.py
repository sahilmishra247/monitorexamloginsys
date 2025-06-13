
# voiceauth.py
from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import base64
import numpy as np
import os
import librosa
from resemblyzer import VoiceEncoder
import io
from typing import Optional
import json

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Pydantic models for API requests
class VoiceAuthRequest(BaseModel):
    username: str
    auth_method: str
    voice_data: str

class AuthResponse(BaseModel):
    success: bool
    message: str

# Constants
SAMPLE_RATE = 16000
VOICE_THRESHOLD = 0.80
EMBEDDING_DIR = "embeddings"
FINGERPRINT_DIR = "fingerprints"
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(FINGERPRINT_DIR, exist_ok=True)

# Static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper functions for voice processing
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

def extract_voice_embedding(y: np.ndarray) -> np.ndarray:
    encoder = VoiceEncoder()
    return encoder.embed_utterance(y)

# Main route to serve the HTML page (serve the frontend HTML file)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML file"""
    try:
        with open("templates\\index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend HTML file not found</h1>", status_code=404)

# New API endpoints that match the frontend expectations
@app.post("/api/register")
async def api_register(request: Request):
    """Handle registration requests from the frontend"""
    try:
        # Check content type to handle both JSON and form data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # Handle JSON request (voice authentication)
            body = await request.json()
            username = body.get("username")
            auth_method = body.get("auth_method")
            voice_data = body.get("voice_data")
            
            if not username or not username.strip():
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Username is required"}
                )
            
            if auth_method == "voice":
                if not voice_data:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": "Voice recording is required"}
                    )
                
                # Process voice authentication
                try:
                    audio_bytes = decode_base64(voice_data)
                    wav_io = io.BytesIO(audio_bytes)

                    print(f"Decoded audio bytes for user '{username}' with length: {len(audio_bytes)}")

                    y, _ = librosa.load(wav_io, sr=SAMPLE_RATE)

                    print(f"Loaded audio for user '{username}' with shape: {y.shape}")

                    y = preprocess_audio(y)
                    voice_embedding = extract_voice_embedding(y)

                    print(f"Extracted voice embedding for user '{username}': {voice_embedding.shape}")

                    voice_embed_path = os.path.join(EMBEDDING_DIR, f"{username}.npy")
                    
                    # Check if user already exists
                    if os.path.exists(voice_embed_path):
                        return JSONResponse(
                            status_code=409,
                            content={"success": False, "message": f"User '{username}' is already registered"}
                        )
                    
                    # Save voice embedding
                    np.save(voice_embed_path, voice_embedding)

                    print(f"Saved voice embedding for user '{username}' at {voice_embed_path}")

                    return JSONResponse(content={
                        "success": True, 
                        "message": f"User '{username}' registered successfully with voice authentication!"
                    })
                    
                except Exception as e:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"Error processing voice data: {str(e)}"}
                    )
            
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Invalid authentication method"}
                )
        
        elif "multipart/form-data" in content_type:
            # Handle form data (fingerprint authentication)
            form = await request.form()
            username = form.get("username")
            auth_method = form.get("auth_method")
            fingerprint = form.get("fingerprint")
            
            if not username or not username.strip():
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Username is required"}
                )
            
            if auth_method == "fingerprint":
                # Fingerprint authentication not implemented yet
                return JSONResponse(
                    status_code=501,
                    content={"success": False, "message": "Fingerprint authentication is not yet implemented. Please use voice authentication."}
                )
            
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Invalid authentication method"}
                )
        
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Unsupported content type"}
            )
    
    except Exception as e:

        print(f"Registration error: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Server error: {str(e)}"}
        )

@app.post("/api/login")
async def api_login(request: Request):
    """Handle login requests from the frontend"""
    try:
        # Check content type to handle both JSON and form data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # Handle JSON request (voice authentication)
            body = await request.json()
            username = body.get("username")
            auth_method = body.get("auth_method")
            voice_data = body.get("voice_data")
            
            if not username or not username.strip():
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Username is required"}
                )
            
            if auth_method == "voice":
                if not voice_data:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": "Voice recording is required"}
                    )
                
                # Process voice authentication
                try:
                    voice_embed_path = os.path.join(EMBEDDING_DIR, f"{username}.npy")
                    
                    # Check if user exists
                    if not os.path.exists(voice_embed_path):
                        return JSONResponse(
                            status_code=404,
                            content={"success": False, "message": f"No voice registration found for user '{username}'"}
                        )
                    
                    # Load stored voice embedding
                    stored_voice = np.load(voice_embed_path)
                    
                    # Process current voice sample
                    audio_bytes = decode_base64(voice_data)
                    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
                    y = preprocess_audio(y)
                    voice_embedding = extract_voice_embedding(y)
                    
                    # Compare voice embeddings
                    voice_similarity = cosine_similarity(voice_embedding, stored_voice)
                    
                    # Check if similarity meets threshold
                    if voice_similarity >= VOICE_THRESHOLD:
                        return JSONResponse(content={
                            "success": True,
                            "message": f"Voice login successful for '{username}'! (Similarity: {voice_similarity:.3f})"
                        })
                    else:
                        return JSONResponse(
                            status_code=401,
                            content={
                                "success": False,
                                "message": f"Voice authentication failed for '{username}' (Similarity: {voice_similarity:.3f})"
                            }
                        )
                        
                except Exception as e:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"Error processing voice data: {str(e)}"}
                    )
            
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Invalid authentication method"}
                )
        
        elif "multipart/form-data" in content_type:
            # Handle form data (fingerprint authentication)
            form = await request.form()
            username = form.get("username")
            auth_method = form.get("auth_method")
            fingerprint = form.get("fingerprint")
            
            if not username or not username.strip():
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Username is required"}
                )
            
            if auth_method == "fingerprint":
                # Fingerprint authentication not implemented yet
                return JSONResponse(
                    status_code=501,
                    content={"success": False, "message": "Fingerprint authentication is not yet implemented. Please use voice authentication."}
                )
            
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "Invalid authentication method"}
                )
        
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Unsupported content type"}
            )
    
    except Exception as e:
        print(f"Login error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Server error: {str(e)}"}
        )

# Legacy form-based endpoint (kept for backward compatibility)
@app.post("/")
async def handle_auth(
    request: Request,
    action: str = Form(...),
    username: str = Form(...),
    auth_method: str = Form(...),
    fingerprint: Optional[UploadFile] = File(None),
    voice_data: Optional[str] = Form(None)
):
    """Legacy form-based endpoint - kept for backward compatibility"""
    try:
        if not username.strip():
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": "Username is required"
            })
        
        # Only process voice authentication (fingerprint is ignored)
        if auth_method == "voice":
            if not voice_data:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "message": "Voice recording is required for voice authentication"
                })
            
            # Process voice
            audio_bytes = decode_base64(voice_data)
            y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
            y = preprocess_audio(y)
            voice_embedding = extract_voice_embedding(y)
            
            # File path for voice
            voice_embed_path = os.path.join(EMBEDDING_DIR, f"{username}.npy")
            
            if action == "register":
                # Check if user already exists
                if os.path.exists(voice_embed_path):
                    return templates.TemplateResponse("index.html", {
                        "request": request,
                        "message": f"User '{username}' is already registered"
                    })
                
                # Save voice embedding
                np.save(voice_embed_path, voice_embedding)
                
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "message": f"✅ User '{username}' registered successfully with voice authentication!"
                })
            
            elif action == "login":
                # Check if user exists
                if not os.path.exists(voice_embed_path):
                    return templates.TemplateResponse("index.html", {
                        "request": request,
                        "message": f"❌ No voice registration found for user '{username}'"
                    })
                
                # Load stored voice
                stored_voice = np.load(voice_embed_path)
                
                # Compare voice
                voice_similarity = cosine_similarity(voice_embedding, stored_voice)
                
                # Voice authentication
                if voice_similarity >= VOICE_THRESHOLD:
                    return templates.TemplateResponse("index.html", {
                        "request": request,
                        "message": f"✅ Voice login successful for '{username}'! (Similarity: {voice_similarity:.3f})"
                    })
                else:
                    return templates.TemplateResponse("index.html", {
                        "request": request,
                        "message": f"❌ Voice authentication failed for '{username}' (Similarity: {voice_similarity:.3f})"
                    })
        
        elif auth_method == "fingerprint":
            # Fingerprint method selected but not implemented
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": "🚧 Fingerprint authentication is not yet implemented. Please use voice authentication."
            })
        
        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": "Invalid authentication method selected"
            })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"❌ Error processing request: {str(e)}"
        })

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "Voice authentication service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)