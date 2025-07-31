from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import os
import shutil
import tempfile

app = FastAPI()

# Load encoder and embeddings
encoder = VoiceEncoder()
person_embeddings = np.load("person_embeddings.npy", allow_pickle=True).item()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/")
def root():
    return {"message": "Speaker identification API is running"}

@app.post("/identify-speaker/")
async def identify_speaker(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        wav = preprocess_wav(tmp_path)
        test_embed = encoder.embed_utterance(wav)

        results = []
        for person, embed in person_embeddings.items():
            sim = cosine_similarity(test_embed, embed)
            results.append((person, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = results[0]

        if best_score > 0.75:
            return JSONResponse(content={"match": best_match, "score": round(best_score, 2)})
        else:
            return JSONResponse(content={"match": None, "message": "No confident match found."})
    finally:
        os.remove(tmp_path)
