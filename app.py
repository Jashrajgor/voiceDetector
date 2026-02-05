from fastapi import FastAPI, Header, HTTPException
import base64
import librosa
import numpy as np
import pickle
import soundfile as sf
import uuid
from pydantic import BaseModel

class VoiceRequest(BaseModel):
    audio_base64: str
    audio_format: str


API_KEY = "HACKATHON_SECRET_KEY"

app = FastAPI(title="AI Voice Detection API")

model = pickle.load(open("model.pkl", "rb"))

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return np.array([
        np.mean(librosa.feature.spectral_flatness(y=y)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.mfcc(y=y, sr=sr))
    ]).reshape(1, -1)

@app.post("/detect-voice")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    audio_base64 = request.audio_base64
    audio_format = request.audio_format

    # remove data:audio/...;base64, if present
    if "," in audio_base64:
        audio_base64 = audio_base64.split(",")[1]

    audio_bytes = base64.b64decode(audio_base64)
    file_name = f"{uuid.uuid4()}.{audio_format}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    features = extract_features(file_name)
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])

    return {
        "prediction": "AI Generated Voice" if prediction == 1 else "Human Voice",
        "confidence": round(float(confidence), 2)
    }
