"""Dienst klonen: Stimmklon mit Qwen3-TTS Base (bf16) aus Referenzaudio und Referenztext.

Das Modell hält keinen Zustand je Stimme; die Referenz geht bei jedem Aufruf mit. Schnittstelle wie beim bisherigen
Sprachdienst (`POST /tts/clone`, Formular mit Datei `ref_audio`; `GET /tts/audio/<datei>`).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile

from .ausgabe import Audioergebnis, abholen_anbieten, ablegen
from .unterbau import Einstellungen, Rechner, anwendung, datei_annehmen, wegraeumen
from .laenge import hoechstens_token
from .synthese import mit_tempo, modell_audio

log = logging.getLogger("klangwerk.klonen")
PORT = 8782
VORGABE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
VERSUCHE = 3


class Motor:
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

    def erzeugen(self, **kwargs: Any) -> np.ndarray | None:
        return modell_audio(self.m, **kwargs)


def hauptmodell(e: Einstellungen) -> str:
    """Das Modell dieses Dienstes: KLANGWERK_MODELL der Dienstdatei, sonst die Vorgabe."""
    return e.modell or VORGABE


def modelle(e: Einstellungen) -> list[str]:
    """Die Repos, die der Dienst lädt (Modellprüfung ohne Laden, MR-11)."""
    return [hauptmodell(e)]


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = hauptmodell(e)
    motor = motor or Motor(modell)
    rechner = Rechner("klonen", modell, e)
    app = anwendung("klonen", rechner, motor.laden, "Stimmklon")
    abholen_anbieten(app, e.ergebnisse)

    @app.post("/tts/clone")
    async def klonen(text: str = Form(...), ref_audio: UploadFile = File(...), ref_text: str = Form(""),
                     language: str = Form("german"), speed: float = Form(1.0),
                     modus: str = Form("auto")) -> Audioergebnis:
        """`modus`: "icl" klont mit Referenztext (übernimmt Klang, Sprache und Sprechweise der Probe), "abdruck" nur mit
        dem Stimmabdruck der Probe (Klang ohne Akzent der Probensprache), "auto" mit Text ICL, ohne Abdruck. Ein leerer
        Text darf nie ICL auslösen: das Modell klont dann mit einem falschen Prompt."""
        text = (text or "").strip()
        if not text:
            raise HTTPException(400, "Text darf nicht leer sein")
        if modus not in ("auto", "icl", "abdruck"):
            raise HTTPException(400, "Modus ist auto, icl oder abdruck")
        mit_text = (ref_text or "").strip()
        if modus == "icl" and not mit_text:
            raise HTTPException(400, "Für icl braucht es den Referenztext")
        prompt_text = mit_text if modus == "icl" or (modus == "auto" and mit_text) else None
        if not 0.1 <= speed <= 4.0:
            raise HTTPException(400, "Speed muss zwischen 0.1 und 4.0 liegen")
        referenz = await datei_annehmen(ref_audio, e, "Leere Referenz-Audio-Datei")
        try:
            audio = None
            for versuch in range(VERSUCHE):
                audio = await rechner.rechnen(motor.erzeugen, text=text, ref_audio=str(referenz), ref_text=prompt_text,
                                              language=language, speed=speed, max_tokens=hoechstens_token(text))
                if audio is not None:
                    break
                log.warning("Klon-Versuch %d: kein Audio, wiederhole", versuch + 1)
            if audio is None:
                raise HTTPException(500, "Kein Audio erzeugt")
            # Ein kurzer Sprechbeginn mit Pause kann ein gewolltes Wort sein. Keine Inhaltskorrektur nach Lautstärke.
            audio = await mit_tempo(audio, motor.sample_rate, speed)
            return ablegen(audio, motor.sample_rate, e.ergebnisse, "clone", language=language)
        finally:
            wegraeumen(referenz)

    return app
