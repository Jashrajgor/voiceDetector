from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
import base64
import uuid
import os
import pickle
import librosa
import numpy as np

app = FastAPI(title="AI Voice Detection API")

# API Key (Render env OR fallback)
API_KEY = os.getenv("API_KEY", "HACKATHON_SECRET_KEY")


model = pickle.load(open("model.pkl", "rb"))

# ---------- Request Body Model (CRITICAL) ----------
class VoiceRequest(BaseModel):
    audio_base64: str
    audio_format: str



def extract_features(file_path: str):
    try:
        y, sr = librosa.load(file_path, sr=16000)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported audio file"
        )

    return np.array([
        np.mean(librosa.feature.spectral_flatness(y=y)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.mfcc(y=y, sr=sr))
    ]).reshape(1, -1)



from pydantic import BaseModel

class VoiceRequest(BaseModel):
    audio_base64: str
    audio_format: str
@app.get("/detect-voice")
def detect_voice_get():
    return {
        "status": "ok",
        "message": "Use POST method with audio_base64 and audio_format"
    }
@app.post("/detect-voice")
def detect_voice_post(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    audio_base64 = data.audio_base64
    audio_format = data.audio_format.lower()

    if audio_format not in ["wav", "mp3"]:
        raise HTTPException(status_code=400, detail="Only wav or mp3 supported")

    if "," in audio_base64:
        audio_base64 = audio_base64.split(",")[1]

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    file_name = f"{uuid.uuid4()}.{audio_format}"
    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    features = extract_features(file_name)
    prediction = model.predict(features)[0]
    confidence = float(max(model.predict_proba(features)[0]))

    return {
        "prediction": "AI Generated Voice" if prediction == 1 else "Human Voice",
        "confidence": round(confidence, 2)
    }
