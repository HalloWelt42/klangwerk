"""Dienst hoeren-moss: Spracherkennung mit Sprechertrennung in einem Durchgang, MOSS-Transcribe-Diarize 0.9B (torch).

Das Modell (Whisper-Medium als Hörteil, Qwen3-0.6B als Textteil) schreibt eine ganze Aufnahme als Folge
`[Beginn][S01]Text[Ende]`: Zeiten in Sekunden und eine Sprecherkennung je Beitrag. Es hört lange Aufnahmen am Stück
(keine eigenen Fenster wie beim Dienst hoeren); Wortzeiten gibt es nicht, nur Beitragszeiten.

Schnittstelle wie der Dienst hoeren (`POST /stt`, `POST /v1/audio/transcriptions`), damit eine App den Hördienst nur
über seine Kennung wählt. Unterschiede im Ergebnis: `words` bleibt leer, jedes Segment nennt seinen `speaker` (S01,
S02, ...), `speakers` zählt sie auf. `language` nimmt der Dienst an, das Modell erkennt die Sprache aber selbst.

Hinweis (`prompt`): Text des Aufrufers, etwa Fachwörter; er geht als Hotword-Hinweis hinter die Anweisung des Modells
(so beschreibt es die Herkunft). Wiederholt das Modell nur den Hinweis, fällt der Beitrag weg.

Rechnen auf der Grafik des Macs über torch (mps). Was mps nicht kann, rechnet torch auf dem Prozessor
(PYTORCH_ENABLE_MPS_FALLBACK in der Dienstdatei). Genauigkeit über KLANGWERK_GENAUIGKEIT, Vorgabe float32 wie in der
Herkunft für Rechner ohne cuda.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .hoeren import dauer_s, hinweis_pruefen, sprachcode
from .unterbau import Einstellungen, Rechner, anwendung, datei_annehmen, wegraeumen

log = logging.getLogger("klangwerk.hoeren_moss")
PORT = 8787
VORGABE = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
NAMEN = {"OpenMOSS-Team/MOSS-Transcribe-Diarize": "MOSS-Transcribe-Diarize 0.9B"}
# Hotwords hängen mit ihrem Stichwort hinter die Anweisung der Herkunft (Zeitmarken und Sprecher je Beitrag)
HOTWORDS = "\u70ed\u8bcd\u63d0\u793a\uff1a"
TOKEN_JE_S = 12                      # Obergrenze der erzeugten Zeichen je Sekunde Ton (gesprochen sind es 4 bis 8)
TOKEN_MINDESTENS = 512


def anweisung(hinweis: str) -> str:
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT     # die Anweisung der Herkunft, eine Wahrheit
    return f"{DEFAULT_PROMPT}{HOTWORDS}{hinweis}" if hinweis else DEFAULT_PROMPT


def hoechstens_token(dauer: float) -> int:
    return max(TOKEN_MINDESTENS, math.ceil(dauer * TOKEN_JE_S))


def _vergleich(text: str) -> str:
    return " ".join("".join(z for z in text.lower() if z.isalnum() or z.isspace()).split())


@dataclass(frozen=True)
class Beitrag:
    start: float
    ende: float
    sprecher: str | None
    text: str


def segmente(beitraege: list[Beitrag], hinweis: str = "") -> list[dict]:
    """Beiträge des Modells im Format des Dienstes hoeren: leere, nur wiederholte und reine Hinweis-Beiträge fallen weg,
    Zeiten aufsteigend, Ende nie vor dem Beginn."""
    aus, gesehen, hv = [], set(), _vergleich(hinweis)
    for b in sorted(beitraege, key=lambda x: (x.start, x.ende)):
        text = b.text.strip()
        if not text:
            continue
        if hv and len(_vergleich(text)) > 3 and _vergleich(text) in hv:
            continue
        schluessel = (round(b.start, 2), text)
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        start = round(max(0.0, b.start), 3)
        aus.append({"start": start, "end": round(max(b.ende, start), 3), "text": text, "words": [], "speaker": b.sprecher})
    return aus


def sprecherliste(segs: list[dict]) -> list[str]:
    return sorted({s["speaker"] for s in segs if s.get("speaker")})


class Motor:
    """Das Modell in torch. `erkennen` liefert die Beiträge einer ganzen Datei."""

    def __init__(self, modell: str, genauigkeit: str) -> None:
        self.modell, self.genauigkeit = modell, genauigkeit
        self.geraet: Any = None
        self.dtype: Any = None
        self.netz: Any = None
        self.prozessor: Any = None

    def laden(self) -> bool:
        import torch
        from moss_transcribe_diarize import MossTranscribeDiarizeForConditionalGeneration, MossTranscribeDiarizeProcessor
        from moss_transcribe_diarize.inference_utils import dtype_from_name
        self.geraet = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype_from_name(self.genauigkeit) if self.geraet.type != "cpu" else torch.float32
        netz = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(self.modell, dtype=self.dtype,
                                                                            attn_implementation="sdpa")
        self.netz = netz.to(self.geraet).eval()
        self.prozessor = MossTranscribeDiarizeProcessor.from_pretrained(self.modell)
        log.info("MOSS-Transcribe-Diarize auf %s in %s", self.geraet, self.genauigkeit)
        return True

    def erkennen(self, pfad: str, hinweis: str = "") -> list[Beitrag]:
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription
        ergebnis = generate_transcription(self.netz, self.prozessor, build_transcription_messages(pfad, anweisung(hinweis)),
                                          max_new_tokens=hoechstens_token(dauer_s(pfad)), do_sample=False,
                                          device=self.geraet, dtype=self.dtype)
        self._leeren()
        return [Beitrag(float(s.start), float(s.end), s.speaker, s.text) for s in parse_transcript(ergebnis["text"])]

    @staticmethod
    def _leeren() -> None:
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = e.modell or VORGABE
    name = NAMEN.get(modell, modell)
    motor = motor or Motor(modell, e.genauigkeit)
    rechner = Rechner("hoeren-moss", modell, e)
    app = anwendung("hoeren-moss", rechner, motor.laden, "Hören (MOSS)")

    async def ganz(pfad: Path, sprache: str | None, hinweis: str = "") -> dict:
        beitraege = await rechner.rechnen(motor.erkennen, str(pfad), hinweis)
        segs = segmente(beitraege, hinweis)
        return {"text": " ".join(s["text"] for s in segs), "segments": segs, "speakers": sprecherliste(segs),
                "language": sprachcode(sprache), "model": name}

    @app.post("/stt")
    async def stt(file: UploadFile = File(...), language: str = Form("german"), diarize: bool = Form(True),
                  prompt: str = Form("")):
        hinweis = hinweis_pruefen(prompt)
        pfad = await datei_annehmen(file, e)
        try:
            return await ganz(pfad, language, hinweis)
        finally:
            wegraeumen(pfad)

    @app.post("/v1/audio/transcriptions")
    async def transkription(file: UploadFile = File(...), model: str | None = Form(None), language: str | None = Form(None),
                            response_format: str = Form("json"), prompt: str | None = Form(None)):
        if response_format not in ("json", "text", "verbose_json"):
            raise HTTPException(400, "response_format: json, text oder verbose_json")
        hinweis = hinweis_pruefen(prompt)
        pfad = await datei_annehmen(file, e)
        try:
            ergebnis = await ganz(pfad, language, hinweis)
        finally:
            wegraeumen(pfad)
        if response_format == "text":
            return PlainTextResponse(ergebnis["text"])
        if response_format == "json":
            return {"text": ergebnis["text"]}
        return {"text": ergebnis["text"], "language": ergebnis["language"], "segments": ergebnis["segments"]}

    return app
