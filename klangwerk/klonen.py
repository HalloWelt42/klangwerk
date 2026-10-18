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

log = logging.getLogger("klangwerk.klonen")
PORT = 8782
VORGABE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
VERSUCHE = 3


def artefakt_entfernen(audio: np.ndarray, sr: int, thresh_db: float = -35.0, min_gap: float = 0.13,
                       max_blip: float = 0.45, max_look: float = 1.1) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Entfernt ein kurzes führendes Klon-Artefakt: Qwen3-TTS hängt beim Klonen am Satzanfang oft einen kurzen,
    referenzbedingten Laut an, durch eine Pause von der Rede getrennt (klingt wie ein zusätzliches Wort).

    Bewusst vorsichtig, damit nie echte Rede verloren geht: geschnitten wird nur, wenn das Audio mit Klang beginnt,
    dieser Laut kurz ist (höchstens max_blip), eine echte Pause folgt (mindestens min_gap) und die Rede spätestens bei
    max_look beginnt. Unverändert übernommen aus dem bisherigen Sprachdienst."""
    a = audio if audio.ndim == 1 else audio.mean(axis=1)
    n = len(a)
    if n == 0:
        return audio, None
    hop, win = max(1, int(0.01 * sr)), max(1, int(0.02 * sr))
    idx = np.arange(0, n, hop)
    energie = np.array([float(np.sqrt(np.mean(a[i:i + win].astype(np.float64) ** 2))) if i < n else 0.0 for i in idx])
    spitze = float(energie.max()) if energie.size else 0.0
    if spitze <= 0:
        return audio, None
    schwelle = max(spitze * (10 ** (thresh_db / 20.0)), spitze * 0.03)
    klang = energie > schwelle
    if not klang[0]:
        return audio, None
    k = 0
    while k < len(klang) and klang[k]:
        k += 1
    laut_ende = idx[k] / sr if k < len(klang) else None
    if laut_ende is None or laut_ende > max_blip:
        return audio, None
    while k < len(klang) and not klang[k]:
        k += 1
    if k >= len(klang):
        return audio, None
    rede = idx[k] / sr
    if (rede - laut_ende) < min_gap or rede > max_look:
        return audio, None
    return a[int(rede * sr):], (laut_ende, rede)


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
        ergebnisse = list(self.m.generate(**kwargs))
        if not ergebnisse or ergebnisse[0].audio is None:
            return None
        return np.array(ergebnisse[0].audio)


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = e.modell or VORGABE
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
                                              language=language, speed=speed, max_tokens=hoechstens_token(text, speed))
                if audio is not None:
                    break
                log.warning("Klon-Versuch %d: kein Audio, wiederhole", versuch + 1)
            if audio is None:
                raise HTTPException(500, "Kein Audio erzeugt")
            audio, geschnitten = artefakt_entfernen(audio, motor.sample_rate)
            if geschnitten:
                log.info("Führendes Artefakt entfernt (Laut bis %.3f s, Rede ab %.3f s)", *geschnitten)
            return ablegen(audio, motor.sample_rate, e.ergebnisse, "clone", language=language)
        finally:
            wegraeumen(referenz)

    return app
