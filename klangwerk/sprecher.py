"""Dienst sprecher: Stimmabdruck je Zeitabschnitt mit ECAPA-TDNN (SpeechBrain), 192 Werte, L2-normiert.

Läuft bewusst auf dem Prozessor: das Modell ist klein, die Abschnitte sind kurz. Abschnitte unter 1 s liefern null
(darunter ist der Abdruck unzuverlässig). Schnittstelle wie bisher: `POST /speaker/embed` mit Datei und `segments`
(JSON-Liste von {start, end}).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile

from .stimmdaten import RATE, mono_16k
from .unterbau import Einstellungen, Rechner, anwendung, datei_annehmen, wegraeumen

PORT = 8784
VORGABE = "speechbrain/spkrec-ecapa-voxceleb"
MINDESTENS_S = 1.0
DIMENSION = 192


class Motor:
    def __init__(self, modell: str) -> None:
        self.modell = modell
        self.m: Any = None

    def laden(self) -> Any:
        from speechbrain.inference.speaker import EncoderClassifier
        self.m = EncoderClassifier.from_hparams(source=self.modell, run_opts={"device": "cpu"})
        return self.m

    def abdruecke(self, pfad: str, abschnitte: list[dict]) -> list[list[float] | None]:
        import torch
        wav = mono_16k(pfad)
        aus: list[list[float] | None] = []
        for s in abschnitte:
            von, bis = max(0.0, float(s.get("start", 0.0))), float(s.get("end", 0.0))
            a, b = int(von * RATE), min(len(wav), int(bis * RATE))
            if bis - von < MINDESTENS_S or b - a < int(MINDESTENS_S * RATE):
                aus.append(None)
                continue
            with torch.no_grad():
                e = self.m.encode_batch(torch.from_numpy(wav[a:b]).unsqueeze(0)).squeeze().cpu().numpy()
            n = float(np.linalg.norm(e))
            if n > 0:
                e = e / n
            aus.append([round(float(x), 6) for x in e.tolist()])
        return aus


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = e.modell or VORGABE
    motor = motor or Motor(modell)
    rechner = Rechner("sprecher", modell, e)
    app = anwendung("sprecher", rechner, motor.laden, "Sprechermerkmal")

    @app.post("/speaker/embed")
    async def abdruck(file: UploadFile = File(...), segments: str = Form(...)):
        try:
            abschnitte = json.loads(segments)
            if not isinstance(abschnitte, list):
                raise ValueError
        except ValueError:
            raise HTTPException(400, "segments muss eine JSON-Liste sein") from None
        pfad = await datei_annehmen(file, e)
        try:
            return {"embeddings": await rechner.rechnen(motor.abdruecke, str(pfad), abschnitte), "dim": DIMENSION}
        finally:
            wegraeumen(pfad)

    return app
