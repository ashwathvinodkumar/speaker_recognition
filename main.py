from fastapi import FastAPI, UploadFile, File
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import os
import shutil
from tempfile import NamedTemporaryFile

app = FastAPI()

# Load saved embeddings (already precomputed)
embedding_path = "person_embeddings.npy"
person_embeddings = np.load(embedding_path, allow_pickle=True).item()

# Initialize encoder
encoder = VoiceEncoder()

# Cosine similarity
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/predict/")
async def predict_speaker(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    finally:
        file.file.close()

    try:
        wav = preprocess_wav(tmp_path)
        test_embed = encoder.embed_utterance(wav)

        # Match against stored embeddings
        results = []
        for person, embed in person_embeddings.items():
            sim = cosine_similarity(test_embed, embed)
            results.append((person, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = results[0]

        # Thresholding (optional)
        threshold = 0.75
        if best_score > threshold:
            return {"match": best_match, "score": best_score}
        else:
            return {"match": None, "score": best_score, "message": "No confident match found"}

    finally:
        os.remove(tmp_path)
