from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import os
import shutil
import tempfile
import traceback

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Speaker identification API is running"}

# Load model and embeddings
try:
    encoder = VoiceEncoder()
    person_embeddings = np.load("person_embeddings.npy", allow_pickle=True).item()
except Exception as e:
    print("❌ Error loading model or embeddings:", e)
    traceback.print_exc()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/identify-speaker/")
async def identify_speaker(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        wav = preprocess_wav(tmp_path)
        test_embed = encoder.embed_utterance(wav)

        results = []
        for person, embed in person_embeddings.items():
            sim = cosine_similarity(test_embed, embed)
            results.append((person, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = results[0]

        os.remove(tmp_path)

        if best_score > 0.50:
            return JSONResponse(content={"match": best_match, "score": float(round(best_score, 2))})
        else:
            return JSONResponse(content={"match": None, "message": "No confident match found."})
    
    except Exception as e:
        # Log error and return JSON error response
        error_message = f"❌ Server error: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": error_message})
