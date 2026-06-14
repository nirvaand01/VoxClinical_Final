"""
NeuraSpeech API
===============
FastAPI wrapper around agent_pipeline.run_pipeline() for the VoxClinical frontend.

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from agent_pipeline import run_pipeline

app = FastAPI(title="NeuraSpeech API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),
):
    if audio is None and not text:
        raise HTTPException(400, "Provide an audio file and/or text.")

    audio_path = None
    try:
        if audio is not None:
            suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await audio.read())
                audio_path = tmp.name

        try:
            result = run_pipeline(audio_path=audio_path, text=text or None)
        except Exception as exc:
            raise HTTPException(500, f"Pipeline error: {exc}") from exc
    finally:
        if audio_path:
            os.unlink(audio_path)

    if "error" in result:
        raise HTTPException(422, result["error"])

    return result
