"""Erzeugte Audiodateien: ablegen, abholen, als WAV-Bytes ausliefern (Sprechen und Stimmklon teilen das)."""

from __future__ import annotations

import io
import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

DATEINAME = re.compile(r"^[A-Za-z0-9_.-]+\.wav$")


class Audioergebnis(BaseModel):
    """Antwort wie beim bisherigen Sprachdienst: die Datei holt man mit GET /tts/audio/<filename>."""

    filename: str
    file_size: int
    duration: float
    sample_rate: int
    voice: str | None = None
    language: str | None = None


def ablegen(audio: np.ndarray, sample_rate: int, ordner: Path, kennzeichen: str, voice: str | None = None,
            language: str | None = None) -> Audioergebnis:
    kennzeichen = re.sub(r"[^A-Za-z0-9_-]", "_", kennzeichen)[:40] or "audio"
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kennzeichen}_{uuid.uuid4().hex[:8]}.wav"
    pfad = ordner / name
    sf.write(str(pfad), audio, sample_rate)
    return Audioergebnis(filename=name, file_size=pfad.stat().st_size, duration=round(len(audio) / sample_rate, 2),
                         sample_rate=sample_rate, voice=voice, language=language)


def als_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    puffer = io.BytesIO()
    sf.write(puffer, audio, sample_rate, format="WAV")
    return puffer.getvalue()


def abholen_anbieten(app: FastAPI, ordner: Path) -> None:
    """GET /tts/audio/<name>: nur einfache Dateinamen aus dem Ergebnisordner, nie Pfade."""

    @app.get("/tts/audio/{dateiname}")
    async def audio(dateiname: str):
        if not DATEINAME.match(dateiname) or not (ordner / dateiname).is_file():
            raise HTTPException(404, "Datei nicht gefunden")
        return FileResponse(str(ordner / dateiname), media_type="audio/wav", filename=dateiname)
