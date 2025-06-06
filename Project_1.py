from fastapi import FastAPI, HTTPException, UploadFile, Form, Request
import base64
import numpy as np
from typing import List, Optional

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
