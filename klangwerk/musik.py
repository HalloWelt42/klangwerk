"""Musik: MiniMax-Music3 in MLX (mlx-audio, Gewichte von mlx-community), ein Stück aus Beschreibung und Liedtext
(HELF-131). Läuft wie die übrigen Klangwerk-Dienste im Maschinenraum hinter der Leitstelle (Fähigkeit musik), mit der
harten Speichergrenze des Maschinenraums für MLX. Das Modell kann kein Referenzaudio (keine Fortsetzung, keine
Bearbeitung); dafür bleibt ACE-Step in der Klangschmiede.

POST /musik         Beschreibung, Liedtext (mit [Verse], [Chorus] ...; "[instrumental]" für ohne Gesang), Dauer,
                    Schritte, Startwert; Antwort wie /tts: Datei, Länge, dazu die Rechenzeit
GET /musik/steckbrief  was das Modell kann (Höchstdauer, Schritte, Referenzaudio, Lizenzhinweis), für Apps, die nur
                    Können zeigen
GET /tts/audio/<n>  die erzeugte Datei holen (wie bei den übrigen Klangwerk-Diensten)
GET /gesundheit     bereit, sobald das Modell geladen ist"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .ausgabe import Audioergebnis, abholen_anbieten, ablegen
from .unterbau import Einstellungen, Rechner, anwendung

log = logging.getLogger("klangwerk.musik")
PORT = 8792
VORGABE = "mlx-community/MiniMax-Music3-mxfp8"
#: so lang darf ein Stück werden (das Modell schafft etwa 5 Minuten)
HOECHSTENS_S = 300.0
#: mehr Schritte lässt der MLX-Port nicht zu
SCHRITTE_HOECHSTENS = 30
#: die Lizenz von MiniMax-Music3 verlangt diesen Hinweis gut sichtbar bei erzeugten Stücken
LIZENZHINWEIS = "MiniMax-Music3"


class Musikauftrag(BaseModel):
    beschreibung: str = Field(min_length=1, max_length=4000, description="Stil, Stimmung, Besetzung, Gesang (am besten "
                                                                       "gegliedert: Global, Gesang, Arrangement)")
    liedtext: str = Field(min_length=1, max_length=8000, description='Liedtext mit Abschnittsmarken; "[instrumental]" '
                                                                     'für ein Stück ohne Gesang')
    dauer_s: float | None = Field(default=None, gt=0, le=HOECHSTENS_S, description="Obergrenze der Länge; leer: das "
                                                                                    "Modell entscheidet")
    schritte: int = Field(default=SCHRITTE_HOECHSTENS, ge=4, le=SCHRITTE_HOECHSTENS, description="Schritte des Flow-Teils (empfohlen und höchstens 30, "
                                                                "mehr lässt der MLX-Port nicht zu)")
    startwert: int = Field(default=0, ge=0)


class Musikergebnis(Audioergebnis):
    rechenzeit_s: float


class Musiksteckbrief(BaseModel):
    """Selbstauskunft des Musikdienstes; die Leitstelle reicht sie unter /v1/dienste/<name>/musik/steckbrief durch."""

    modell: str = Field(description="Name des Modells, wie ihn eine Oberfläche zeigt")
    gewichte: str = Field(description="Quelle der Gewichte (Hugging Face), daran die Genauigkeit")
    hoechstdauer_s: float
    schritte_hoechstens: int
    referenzaudio: bool = Field(description="kann ein Stück aus einer Aufnahme fortsetzen, covern oder bearbeiten")
    rate: int
    kanaele: int
    lizenzhinweis: str = Field(description="sichtbarer Hinweis, den die Lizenz bei erzeugten Stücken verlangt")


class Motor:
    def __init__(self, modell: str) -> None:
        self.modell = modell
        self._m: Any = None
        self.sample_rate = 44100

    def laden(self) -> Any:
        from mlx_audio.music.utils import load_model
        self._m = load_model(self.modell)
        return self._m

    def erzeugen(self, beschreibung: str, liedtext: str, dauer_s: float | None, schritte: int, startwert: int) -> np.ndarray:
        art: dict[str, Any] = {"text": beschreibung, "lyrics": liedtext, "steps": schritte, "seed": startwert}
        if dauer_s is not None:
            art["duration"] = dauer_s
        stuecke = []
        for teil in self._m.generate(**art):
            self.sample_rate = teil.sample_rate
            stuecke.append(np.array(teil.audio))
        if not stuecke:
            raise RuntimeError("kein Audio erzeugt")
        return stuecke[0] if len(stuecke) == 1 else np.concatenate(stuecke, axis=0)


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = e.modell or VORGABE
    motor = motor or Motor(modell)
    rechner = Rechner("musik", modell, e)
    app = anwendung("musik", rechner, motor.laden, "Musik")
    abholen_anbieten(app, e.ergebnisse)

    @app.get("/musik/steckbrief")
    async def steckbrief() -> Musiksteckbrief:
        return Musiksteckbrief(modell="MiniMax-Music3", gewichte=modell, hoechstdauer_s=HOECHSTENS_S,
                               schritte_hoechstens=SCHRITTE_HOECHSTENS, referenzaudio=False, rate=motor.sample_rate,
                               kanaele=2, lizenzhinweis=LIZENZHINWEIS)

    @app.post("/musik")
    async def musik(a: Musikauftrag) -> Musikergebnis:
        beginn = time.monotonic()
        try:
            audio = await rechner.rechnen(motor.erzeugen, a.beschreibung.strip(), a.liedtext.strip(), a.dauer_s, a.schritte,
                                          a.startwert)
        except ValueError as f:
            raise HTTPException(422, str(f)) from f
        except RuntimeError as f:
            raise HTTPException(500, str(f)) from f
        rechenzeit = time.monotonic() - beginn
        ergebnis = ablegen(audio, motor.sample_rate, e.ergebnisse, "musik")
        log.info("Stück fertig: %.1f s Musik in %.1f s", ergebnis.duration, rechenzeit)
        return Musikergebnis(**ergebnis.model_dump(), rechenzeit_s=round(rechenzeit, 1))

    return app
