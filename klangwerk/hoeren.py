"""Dienst hoeren: Spracherkennung mit Whisper Large V3 (MLX), dazu Silero für Sprachpausen beim Strom.

Schutz vor Halluzinationen wie im bisherigen Sprachdienst: kein Bezug auf den vorigen Text, Schwellen für Stille und
Kompression, Nachbearbeitung (Wiederholungen und typische Halluzinationen entfernen, Abschnitte von 6 bis 15 s an
Pausen bilden, Wortzeiten erhalten). Sprechertrennung gehört nicht hierher, sondern in den Dienst trennen; der
Formularwert `diarize` wird angenommen und ignoriert (`speakers` bleibt leer).

Schnittstelle wie bisher (`POST /stt`, `POST /stt/stream` als Ereignisstrom) und im üblichen Format
(`POST /v1/audio/transcriptions`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from .unterbau import Einstellungen, Rechner, anwendung, datei_annehmen, wegraeumen

log = logging.getLogger("klangwerk.hoeren")
PORT = 8783
VORGABE = "mlx-community/whisper-large-v3-mlx"
NAMEN = {"mlx-community/whisper-large-v3-mlx": "Whisper Large V3 (MLX)"}
ABSCHNITT_HOECHSTENS_S = 15          # Strom: ab 15 s wird an der nächsten Pause geteilt
SEGMENT_MINDESTENS_S = 6             # Segmente unter 6 s zusammenführen
SEGMENT_HOECHSTENS_S = 15            # über 15 s an einer Pause teilen
PAUSE_S = 0.3
HALLUZINATIONEN = ("Untertitel von", "Untertitelung", "Vielen Dank für", "Copyright", "www.", "http", "Abonniere", "Musik", "♪", "🎵")
SPRACHCODES = {"german": "de", "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko", "french": "fr", "spanish": "es", "auto": None}


def sprachcode(sprache: str | None) -> str | None:
    if not sprache:
        return None
    return SPRACHCODES.get(sprache, sprache)


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def dauer_s(pfad: str) -> float:
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    try:
        r = subprocess.run([ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", pfad], capture_output=True, text=True, timeout=30)
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def ausschneiden(pfad: str, von: float, bis: float, ziel: str) -> None:
    subprocess.run([_ffmpeg(), "-y", "-v", "quiet", "-i", pfad, "-ss", str(von), "-to", str(bis),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", ziel], check=False)


def woerter(roh: list | None) -> list[dict]:
    """Whisper-Wörter zu {word, start, end}; Zeiten gerundet, leere ohne Zeit verworfen."""
    aus = []
    for w in roh or []:
        tok = w.get("word")
        if tok is None:
            tok = w.get("text", "")
        if not (tok or "").strip() or w.get("start") is None:
            continue
        ende = w.get("end")
        aus.append({"word": tok, "start": round(float(w["start"]), 3), "end": round(float(ende if ende is not None else w["start"]), 3)})
    return aus


def segmente_bereinigen(segmente: list) -> list[dict]:
    """Halluzinationen und Wiederholungen entfernen, dann Abschnitte von 6 bis 15 s an Pausen bilden."""
    gefiltert, gesehen = [], set()
    for s in segmente:
        text = (s.get("text") or "").strip()
        if not text or text in gesehen:
            continue
        w = text.split()
        if len(w) > 3 and len({x.lower() for x in w}) / len(w) < 0.3:
            continue
        if any(m in text for m in HALLUZINATIONEN) and len(text) < 50:
            continue
        gesehen.add(text)
        gefiltert.append({"start": s.get("start", 0), "end": s.get("end", 0), "text": text, "words": woerter(s.get("words"))})
    if not gefiltert:
        return gefiltert
    zusammen, akt = [], {**gefiltert[0], "words": list(gefiltert[0]["words"])}
    for s in gefiltert[1:]:
        dauer, luecke = akt["end"] - akt["start"], s["start"] - akt["end"]
        if dauer >= SEGMENT_HOECHSTENS_S or (dauer >= SEGMENT_MINDESTENS_S and luecke >= PAUSE_S):
            zusammen.append(akt)
            akt = {**s, "words": list(s["words"])}
        else:
            akt["end"] = s["end"]
            akt["text"] += " " + s["text"]
            akt["words"].extend(s["words"])
    zusammen.append(akt)
    return zusammen


class Motor:
    def __init__(self, modell: str) -> None:
        self.modell = modell
        self.vad: Any = None

    def laden(self) -> Any:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.get_model(self.modell, mx.float16)      # einmal laden; transcribe nutzt dieses Modell
        try:
            from silero_vad import load_silero_vad
            self.vad = load_silero_vad()
        except Exception as f:                               # ohne Sprachpausen: feste Abschnitte
            log.warning("Sprachpausenerkennung nicht verfügbar: %s", f)
        return True

    def erkennen(self, pfad: str, sprache: str | None) -> dict:
        import mlx_whisper
        return mlx_whisper.transcribe(pfad, path_or_hf_repo=self.modell, language=sprache, condition_on_previous_text=False,
                                      no_speech_threshold=0.6, compression_ratio_threshold=2.4, word_timestamps=True, verbose=False)

    def abschnitte(self, pfad: str) -> list[dict]:
        """Sprachabschnitte über Silero, zu höchstens 15 s zusammengefasst; ohne Silero feste Abschnitte."""
        try:
            from silero_vad import get_speech_timestamps, read_audio
            if self.vad is None:
                raise RuntimeError("keine Sprachpausenerkennung")
            wav = read_audio(pfad, sampling_rate=16000)
            stellen = get_speech_timestamps(wav, self.vad, sampling_rate=16000, min_silence_duration_ms=500, speech_pad_ms=100)
            if not stellen:
                return [{"start": 0.0, "end": dauer_s(pfad)}]
            aus, von, bis = [], stellen[0]["start"] / 16000, stellen[0]["end"] / 16000
            for s in stellen[1:]:
                a, b = s["start"] / 16000, s["end"] / 16000
                if b - von > ABSCHNITT_HOECHSTENS_S:
                    aus.append({"start": von, "end": bis})
                    von, bis = a, b
                else:
                    bis = b
            aus.append({"start": von, "end": bis})
            return aus
        except Exception as f:
            log.warning("Sprachpausen nicht erkannt, feste Abschnitte: %s", f)
            gesamt, aus, ab = dauer_s(pfad), [], 0.0
            while ab < gesamt:
                aus.append({"start": ab, "end": min(ab + ABSCHNITT_HOECHSTENS_S, gesamt)})
                ab += ABSCHNITT_HOECHSTENS_S
            return aus


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = e.modell or VORGABE
    name = NAMEN.get(modell, modell)
    motor = motor or Motor(modell)
    rechner = Rechner("hoeren", modell, e)
    app = anwendung("hoeren", rechner, motor.laden, "Hören")

    async def ganz(pfad: Path, sprache: str) -> dict:
        roh = await rechner.rechnen(motor.erkennen, str(pfad), sprachcode(sprache))
        segmente = segmente_bereinigen(roh.get("segments", []))
        return {"text": " ".join(s["text"].strip() for s in segmente), "segments": segmente, "speakers": [],
                "language": roh.get("language", sprache), "model": name}

    @app.post("/stt")
    async def stt(file: UploadFile = File(...), language: str = Form("german"), diarize: bool = Form(True)):
        pfad = await datei_annehmen(file, e)
        try:
            return await ganz(pfad, language)
        finally:
            wegraeumen(pfad)

    @app.post("/v1/audio/transcriptions")
    async def transkription(file: UploadFile = File(...), model: str | None = Form(None), language: str | None = Form(None),
                            response_format: str = Form("json")):
        if response_format not in ("json", "text", "verbose_json"):
            raise HTTPException(400, "response_format: json, text oder verbose_json")
        pfad = await datei_annehmen(file, e)
        try:
            ergebnis = await ganz(pfad, language or "auto")
        finally:
            wegraeumen(pfad)
        if response_format == "text":
            return PlainTextResponse(ergebnis["text"])
        if response_format == "json":
            return {"text": ergebnis["text"]}
        return {"text": ergebnis["text"], "language": ergebnis["language"], "segments": ergebnis["segments"]}

    async def strom(pfad: Path, sprache: str) -> AsyncIterator[dict]:
        """Abschnittsweise erkennen und jedes Segment sofort mit absoluten Zeiten liefern."""
        gesamt = await asyncio.to_thread(dauer_s, str(pfad))
        abschnitte = await asyncio.to_thread(motor.abschnitte, str(pfad))
        alle, nummer = [], 0
        for a in abschnitte:
            if a["end"] - a["start"] < 0.3:
                continue
            fd, teil = tempfile.mkstemp(suffix=".wav", prefix="klangwerk-")
            os.close(fd)
            try:
                await asyncio.to_thread(ausschneiden, str(pfad), a["start"], a["end"], teil)
                roh = await rechner.rechnen(motor.erkennen, teil, sprachcode(sprache))
                for s in segmente_bereinigen(roh.get("segments", [])):
                    s["start"], s["end"] = round(s["start"] + a["start"], 3), round(s["end"] + a["start"], 3)
                    for w in s.get("words", []):
                        w["start"], w["end"] = round(w["start"] + a["start"], 3), round(w["end"] + a["start"], 3)
                    alle.append(s)
                    yield s
            except HTTPException:
                raise
            except Exception as f:
                yield {"_error": f"Abschnitt {nummer} ({a['start']:.1f}-{a['end']:.1f} s) fehlgeschlagen: {f}"}
            finally:
                wegraeumen(teil)
            nummer += 1
            await asyncio.sleep(0)
        yield {"_stats": {"model": name, "language": sprache, "total_chunks": nummer, "total_duration": round(gesamt, 1)},
               "_text": " ".join(s["text"].strip() for s in alle)}

    @app.post("/stt/stream")
    async def stt_strom(file: UploadFile = File(...), language: str = Form("german")):
        pfad = await datei_annehmen(file, e)

        async def ereignisse():
            try:
                yield f"data: {json.dumps({'type': 'start', 'model': name})}\n\n"
                segmente, beginn, stats = [], time.time(), {}
                async for s in strom(pfad, language):
                    if "_stats" in s:
                        stats = s["_stats"]
                        continue
                    if "_error" in s:
                        yield f"data: {json.dumps({'type': 'warning', 'message': s['_error']})}\n\n"
                        continue
                    segmente.append(s)
                    yield f"data: {json.dumps({'type': 'segment', 'segment': s, 'index': len(segmente) - 1, 'elapsed': round(time.time() - beginn, 1)})}\n\n"
                text = " ".join(s.get("text", "") for s in segmente)
                yield f"data: {json.dumps({'type': 'done', 'total_segments': len(segmente), 'text': text, 'elapsed': round(time.time() - beginn, 1), 'stats': stats})}\n\n"
            except Exception as f:
                log.exception("Strom abgebrochen")
                yield f"data: {json.dumps({'type': 'error', 'message': str(f)})}\n\n"
            finally:
                wegraeumen(pfad)

        return StreamingResponse(ereignisse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
