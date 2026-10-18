"""Dienst sprechen: Sprachsynthese mit genau einem Qwen3-TTS-Modell (Vorgabe CustomVoice 8-bit, 9 Stimmen).

Ein anderes Synthesemodell (bf16, VoiceDesign) ist ein eigener Dienst mit KLANGWERK_MODELL in seiner Dienstdatei;
so bleibt je Prozess genau ein Modell im Speicher.

Schnittstelle wie beim bisherigen Sprachdienst (`POST /tts`, `GET /tts/audio/<datei>`, `GET /tts/voices`) und im
üblichen Format (`POST /v1/audio/speech`, Antwort WAV).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .ausgabe import Audioergebnis, abholen_anbieten, ablegen, als_wav
from .unterbau import Einstellungen, Rechner, anwendung
from .laenge import hoechstens_token

log = logging.getLogger("klangwerk.sprechen")
PORT = 8781
VORGABE = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
# Kurznamen aus dem bisherigen Sprachdienst, damit Aufrufer mit `model` weiter funktionieren
KURZNAMEN = {
    "1.7b-customvoice-8bit": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "1.7b-customvoice-bf16": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "1.7b-voicedesign-4bit": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
}
VERSUCHE = 3          # Qwen3-TTS liefert sporadisch kein Audio (vor allem bei kurzen Sätzen)


class Sprechauftrag(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "aiden"
    language: str = "german"
    speed: float = Field(1.0, ge=0.1, le=4.0)
    model: str | None = None
    instruct: str | None = None
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=200)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.05, ge=0.5, le=2.0)


class SpeechAnfrage(BaseModel):
    """Übliches Format für Sprachausgabe; dazu die Regler von Qwen3-TTS."""

    input: str = Field(..., min_length=1)
    voice: str = "aiden"
    model: str | None = None
    response_format: str = "wav"
    speed: float = Field(1.0, ge=0.1, le=4.0)
    language: str = "german"
    instruct: str | None = None
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=200)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.05, ge=0.5, le=2.0)


class Motor:
    """Der eigentliche Modellaufruf; in Tests durch einen Schein ersetzt."""

    def __init__(self, modell: str) -> None:
        self.modell = modell
        self.m: Any = None

    def laden(self) -> Any:
        from mlx_audio.tts.utils import load_model
        self.m = load_model(self.modell)
        return self.m

    @property
    def sample_rate(self) -> int:
        return int(self.m.sample_rate)

    def stimmen(self) -> list[str]:
        try:
            return sorted(self.m.get_supported_speakers())
        except Exception:
            return []

    def sprachen(self) -> list[str]:
        try:
            return list(self.m.get_supported_languages())
        except Exception:
            return ["german", "english"]

    def erzeugen(self, **kwargs: Any) -> np.ndarray | None:
        ergebnisse = list(self.m.generate(**kwargs))
        if not ergebnisse or ergebnisse[0].audio is None:
            return None
        return np.array(ergebnisse[0].audio)


def _modell_passt(gewuenscht: str | None, gehalten: str) -> None:
    if gewuenscht and KURZNAMEN.get(gewuenscht, gewuenscht) != gehalten:
        raise HTTPException(409, f"Dieser Dienst hält {gehalten}; für {gewuenscht} einen eigenen Dienst mit diesem Modell starten.")


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = KURZNAMEN.get(e.modell or "", e.modell) or VORGABE
    motor = motor or Motor(modell)
    rechner = Rechner("sprechen", modell, e)
    app = anwendung("sprechen", rechner, motor.laden, "Sprechen")
    abholen_anbieten(app, e.ergebnisse)

    async def synthese(text: str, **regler: Any) -> np.ndarray:
        for versuch in range(VERSUCHE):
            audio = await rechner.rechnen(motor.erzeugen, text=text, max_tokens=hoechstens_token(text, regler.get("speed", 1.0)),
                                          **regler)
            if audio is not None:
                return audio
            log.warning("Synthese-Versuch %d: kein Audio, wiederhole", versuch + 1)
        raise HTTPException(500, "Kein Audio erzeugt")

    def regler(a: Sprechauftrag | SpeechAnfrage) -> dict[str, Any]:
        r = dict(voice=a.voice, language=a.language, speed=a.speed, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                 repetition_penalty=a.repetition_penalty)
        if a.instruct and a.instruct.strip():
            r["instruct"] = a.instruct.strip()
        return r

    @app.post("/tts")
    async def tts(a: Sprechauftrag) -> Audioergebnis:
        text = a.text.strip()
        if not text:
            raise HTTPException(400, "Text darf nicht leer sein")
        _modell_passt(a.model, modell)
        audio = await synthese(text, **regler(a))
        return ablegen(audio, motor.sample_rate, e.ergebnisse, a.voice, voice=a.voice, language=a.language)

    @app.post("/v1/audio/speech")
    async def speech(a: SpeechAnfrage) -> Response:
        if a.response_format != "wav":
            raise HTTPException(400, "Nur response_format wav wird geliefert.")
        _modell_passt(a.model, modell)
        audio = await synthese(a.input.strip(), **regler(a))
        return Response(als_wav(audio, motor.sample_rate), media_type="audio/wav")

    @app.get("/tts/voices")
    async def stimmen():
        if rechner.zustand != "bereit":
            raise HTTPException(503, "Modell lädt noch")
        return {"speakers": motor.stimmen(), "languages": motor.sprachen()}

    @app.get("/tts/models")
    async def modelle():
        kurz = next((k for k, v in KURZNAMEN.items() if v == modell), modell)
        return [{"id": kurz, "name": modell, "hf_id": modell, "installed": True, "geladen": rechner.zustand == "bereit"}]

    return app
