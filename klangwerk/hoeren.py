"""Dienst hoeren: Spracherkennung mit Whisper Large V3 (MLX), dazu Silero für Sprachpausen beim Strom.

Schutz vor Halluzinationen wie im bisherigen Sprachdienst: kein Bezug auf den vorigen Text, Schwellen für Stille und
Kompression, Nachbearbeitung (Wiederholungen und typische Halluzinationen entfernen, Abschnitte von 6 bis 15 s an
Pausen bilden, Wortzeiten erhalten).

Hinweis (`prompt`): ein Text, den der Aufrufer mitgibt, etwa ein Satz in sauberer Schreibweise und seine Fachwörter.
Whisper nimmt ihn als vorigen Text; weil es ohne Bezug auf den vorigen Text hört, gälte er nur im ersten Fenster von
30 s. Mit Hinweis hört der Dienst darum selbst in Fenstern bis 28 s an Sprachpausen und gibt jedem Fenster denselben
Hinweis: Schreibweise und Fachwörter tragen durch die ganze Aufnahme, ohne dass ein Fenster vom vorigen abhängt (ohne
Hinweis kippt Whisper in langen Strecken in Kleinschrift ohne Satzzeichen). Wiederholt Whisper in einer Stille nur den
Hinweis, fällt das Segment weg. Sprechertrennung gehört nicht hierher, sondern in den Dienst trennen; der
Formularwert `diarize` wird angenommen und ignoriert (`speakers` bleibt leer).

Nichts Gesprochenes fällt still weg (BT-91): nach dem Erkennen vergleicht der Dienst die Sprachstellen mit den Wörtern
der Abschrift (Modul `luecken`). Was fehlt, weil Whisper es ausließ oder ein Filter strich, hört er noch einmal für sich,
zuerst in Fenstern bis 12 s, dann bis 6 s ohne Hinweis, zuletzt mit ein paar Sekunden Umfeld, aus dem nur die Wörter
in der Lücke gelten; übernommen wird nur, was Whisper dabei sicher hört. Bleibt dann noch Sprache ohne Text, steht sie in
`luecken`, erfunden wird nichts; `nachgehoert` nennt, was dazukam.

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
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from . import luecken
from .abschnitte import Sprachpausen, planen
from .stimmdaten import RATE, mono_16k
from .unterbau import Einstellungen, Rechner, anwendung, datei_annehmen, wegraeumen

log = logging.getLogger("klangwerk.hoeren")
PORT = 8783
VORGABE = "mlx-community/whisper-large-v3-mlx"
# Varianten als weitere Dienste mit KLANGWERK_MODELL (Dienstdateien klangwerk-hoeren-8bit, klangwerk-hoeren-turbo)
NAMEN = {"mlx-community/whisper-large-v3-mlx": "Whisper Large V3 (MLX)",
         "mlx-community/whisper-large-v3-mlx-8bit": "Whisper Large V3 (8 Bit)",
         "mlx-community/whisper-large-v3-turbo": "Whisper Large V3 Turbo"}
ABSCHNITT_HOECHSTENS_S = 15          # Strom: ab 15 s wird an der nächsten Pause geteilt
SEGMENT_MINDESTENS_S = 6             # Segmente unter 6 s zusammenführen
SEGMENT_HOECHSTENS_S = 15            # über 15 s an einer Pause teilen
PAUSE_S = 0.3
FENSTER_S = 28                       # mit Hinweis: eigene Fenster unter den 30 s von Whisper, damit jedes ihn bekommt


@dataclass(frozen=True)
class Versuch:
    fenster_s: float              # Fenster höchstens
    mit_hinweis: bool = False
    umfeld_s: float = 0.0         # so viel davor und danach wird mitgehört; von dort gilt kein Wort


NACHHOEREN = (Versuch(12.0, mit_hinweis=True), Versuch(6.0), Versuch(20.0, umfeld_s=4.0))
SICHER_LOGPROB, SICHER_KOMPRESSION, SICHER_OHNE_SPRACHE = -1.0, 2.4, 0.6      # Nachgehörtes: nur was Whisper sicher hört
HINWEIS_HOECHSTENS = 1000            # Zeichen; Whisper nimmt ohnehin nur die letzten 223 Merkmale des vorigen Texts
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


def _vergleich(text: str) -> str:
    return " ".join("".join(z for z in text.lower() if z.isalnum() or z.isspace()).split())


@dataclass(frozen=True)
class Aussortiert:
    start: float
    end: float
    text: str
    grund: str


def _phrase(text: str) -> bool:
    return any(m in text for m in HALLUZINATIONEN) and len(text) < 50


def _einzelnes(s: dict) -> dict:
    return {"start": s.get("start", 0), "end": s.get("end", 0), "text": (s.get("text") or "").strip(), "words": woerter(s.get("words"))}


def aussortieren(segmente: Sequence[dict], hinweis: str = "") -> tuple[list[dict], list[Aussortiert]]:
    """Halluzinationen und Wiederholungen (auch ein nur wiederholter Hinweis) von den übrigen Segmenten trennen."""
    gut, weg, gesehen = [], [], set()
    hinweis_vergleich = _vergleich(hinweis)
    for s in segmente:
        x = _einzelnes(s)
        text, w = x["text"], x["text"].split()
        if not text:
            continue
        grund = ("wiederholt" if text in gesehen
                 else "hinweis" if hinweis_vergleich and len(_vergleich(text)) > 3 and _vergleich(text) in hinweis_vergleich
                 else "schleife" if len(w) > 3 and len({y.lower() for y in w}) / len(w) < 0.3
                 else "phrase" if _phrase(text) else "")
        if grund:
            weg.append(Aussortiert(x["start"], x["end"], text, grund))
            continue
        gesehen.add(text)
        gut.append(x)
    return gut, weg


def sicher(s: dict) -> bool:
    """Hört Whisper das Segment sicher? Ohne Angaben (andere Motoren) ja."""
    if (s.get("avg_logprob") or 0.0) < SICHER_LOGPROB or (s.get("compression_ratio") or 0.0) > SICHER_KOMPRESSION:
        return False
    return (s.get("no_speech_prob") or 0.0) <= SICHER_OHNE_SPRACHE or (s.get("avg_logprob") or 0.0) > -0.5


def nur_in(segmente: Sequence[dict], von: float, bis: float) -> list[dict]:
    """Nur die Wörter, die ganz in dieser Strecke liegen; der Text folgt ihnen."""
    aus = []
    for s in segmente:
        worte = [w for w in s.get("words") or [] if w["start"] >= von and w["end"] <= bis]
        if worte:
            aus.append({**s, "start": worte[0]["start"], "end": worte[-1]["end"], "words": worte, "text": "".join(w["word"] for w in worte)})
    return aus


def nachgehoert_annehmen(segmente: Sequence[dict], hinweis: str, frueher: Sequence[str]) -> list[dict]:
    """Was vom Nachhören bleibt. Eine Wiederholung von früher im Text ist hier erlaubt: zweimal für sich gehört, ist sie
    gesprochen. Eine typische Halluzination gilt nur, wenn schon das erste Hören sie an dieser Stelle hatte."""
    gut, weg = aussortieren([s for s in segmente if sicher(s)], hinweis)
    bekannt = {_vergleich(t) for t in frueher}
    gut += [{"start": a.start, "end": a.end, "text": a.text, "words": _worte_von(segmente, a)} for a in weg
            if a.grund == "phrase" and _vergleich(a.text) in bekannt]
    return sorted(gut, key=lambda s: s["start"])


def _worte_von(segmente: Sequence[dict], a: Aussortiert) -> list[dict]:
    return next((woerter(s.get("words")) for s in segmente if s.get("start") == a.start and (s.get("text") or "").strip() == a.text), [])


def zusammenfassen(segmente: Sequence[dict]) -> list[dict]:
    """Abschnitte von 6 bis 15 s, getrennt an Pausen."""
    if not segmente:
        return []
    zusammen, akt = [], {**segmente[0], "words": list(segmente[0]["words"])}
    for s in segmente[1:]:
        dauer, luecke = akt["end"] - akt["start"], s["start"] - akt["end"]
        if dauer >= SEGMENT_HOECHSTENS_S or (dauer >= SEGMENT_MINDESTENS_S and luecke >= PAUSE_S):
            zusammen.append(akt)
            akt = {**s, "words": list(s["words"])}
        else:
            akt["end"] = max(akt["end"], s["end"])
            akt["text"] += " " + s["text"]
            akt["words"].extend(s["words"])
    zusammen.append(akt)
    return zusammen


def segmente_bereinigen(segmente: Sequence[dict], hinweis: str = "") -> list[dict]:
    return zusammenfassen(aussortieren(segmente, hinweis)[0])


def verschieben(segmente: list, versatz: float) -> list[dict]:
    """Segmente und ihre Wörter um `versatz` Sekunden später (Fenster oder Abschnitt in der ganzen Datei)."""
    aus = []
    for s in segmente:
        woerter_neu = [{**w, "start": w.get("start", 0) + versatz, "end": w.get("end", 0) + versatz} for w in s.get("words") or []
                       if w.get("start") is not None]
        aus.append({**s, "start": s.get("start", 0) + versatz, "end": s.get("end", 0) + versatz, "words": woerter_neu})
    return aus


def abschnitt_pruefen(abschnitt_s: float | None, obergrenze_s: float = 3600.0) -> float | None:
    """Gewünschte Abschnittslänge (Formularwert `abschnitt_s`): leer heißt Vorgabe des Dienstes."""
    if abschnitt_s is None or abschnitt_s == 0:
        return None
    if not 5 <= abschnitt_s <= obergrenze_s:
        raise HTTPException(400, f"abschnitt_s: 5 bis {obergrenze_s:g} Sekunden")
    return float(abschnitt_s)


def hinweis_pruefen(hinweis: str | None) -> str:
    hinweis = (hinweis or "").strip()
    if len(hinweis) > HINWEIS_HOECHSTENS:
        raise HTTPException(400, f"prompt: höchstens {HINWEIS_HOECHSTENS} Zeichen")
    return hinweis


def fensterlaenge(hinweis: str, abschnitt_s: float | None) -> float | None:
    """Länge der eigenen Fenster: mit Hinweis oder gewünschter Abschnittslänge höchstens 28 s (Whisper hört 30 s am
    Stück), sonst keine eigenen Fenster (None: Whisper hört die ganze Datei selbst)."""
    if abschnitt_s:
        return min(float(abschnitt_s), FENSTER_S)
    return FENSTER_S if hinweis else None


class Motor:
    def __init__(self, modell: str) -> None:
        self.modell = modell
        self.pausen = Sprachpausen()

    def laden(self) -> Any:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.get_model(self.modell, mx.float16)      # einmal laden; transcribe nutzt dieses Modell
        self.pausen.laden()                                  # ohne Sprachpausen kein Dienst: nie feste Schnitte
        return True

    def ton(self, pfad: str) -> np.ndarray:
        return mono_16k(pfad)

    def stellen(self, ton: np.ndarray) -> list[luecken.Strecke]:
        return self.pausen.stellen(ton)

    def _hoeren(self, ton: np.ndarray, sprache: str | None, hinweis: str) -> dict:
        import mlx_whisper
        return mlx_whisper.transcribe(ton, path_or_hf_repo=self.modell, language=sprache, condition_on_previous_text=False,
                                      no_speech_threshold=0.6, compression_ratio_threshold=2.4, word_timestamps=True, verbose=False,
                                      **({"initial_prompt": hinweis} if hinweis else {}))

    def in_fenstern(self, ton: np.ndarray, sprache: str | None, hinweis: str, stellen: Sequence[luecken.Strecke], fenster_s: float) -> dict:
        """In eigenen Fenstern an Sprechpausen, jedes mit dem Hinweis; Zeiten auf die Datei."""
        segmente, erkannt = [], None
        plan = planen(stellen, len(ton) / RATE, fenster_s)
        for f in plan:
            stueck = ton[int(f.start * RATE):int(f.ende * RATE)]
            if len(stueck) < 0.3 * RATE:
                continue
            roh = self._hoeren(stueck, sprache, hinweis)
            erkannt = erkannt or roh.get("language")
            segmente += verschieben(roh.get("segments", []), f.start)
        return {"segments": segmente, "language": erkannt or sprache, "abschnitte": [f.als_dict() for f in plan]}

    def erkennen(self, ton: np.ndarray, sprache: str | None, hinweis: str = "", abschnitt_s: float | None = None,
                 stellen: Sequence[luecken.Strecke] = ()) -> dict:
        """Ganze Datei. Mit Hinweis oder gewünschter Abschnittslänge in eigenen Fenstern (siehe oben)."""
        fenster_s = fensterlaenge(hinweis, abschnitt_s)
        if fenster_s is None:
            return {**self._hoeren(ton, sprache, ""), "abschnitte": []}
        return self.in_fenstern(ton, sprache, hinweis, stellen, fenster_s)

    def nachhoeren(self, ton: np.ndarray, von: float, bis: float, sprache: str | None, hinweis: str,
                   stellen: Sequence[luecken.Strecke], fenster_s: float) -> list[dict]:
        """Nur diese Strecke, in kleinen Fenstern; Zeiten auf die Datei."""
        darin = [(max(a, von) - von, min(b, bis) - von) for a, b in stellen if a < bis and b > von]
        return verschieben(self.in_fenstern(ton[int(von * RATE):int(bis * RATE)], sprache, hinweis, darin, fenster_s)["segments"], von)

    def abschnitte(self, pfad: str, hoechstens_s: float = ABSCHNITT_HOECHSTENS_S) -> list[dict]:
        return [{"start": f.start, "end": f.ende} for f in self.pausen.abschnitte(mono_16k(pfad), hoechstens_s)]


def nachholen(motor: Any, ton: np.ndarray, stellen: Sequence[luecken.Strecke], gut: list[dict], weg: list[Aussortiert],
              sprache: str | None, hinweis: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Sprache ohne Text noch einmal hören. Segmente samt Nachgehörtem, was dazukam, und was trotzdem fehlt."""
    gestrichen = [(a.start, a.end, a.grund) for a in weg]
    dazu: list[dict] = []
    for v in NACHHOEREN:
        for l in luecken.finden(stellen, gut, gestrichen):
            h = hinweis if v.mit_hinweis else ""
            roh = motor.nachhoeren(ton, max(0.0, l.start - v.umfeld_s), l.ende + v.umfeld_s, sprache, h, stellen, v.fenster_s)
            if v.umfeld_s:
                roh = nur_in(roh, l.start, l.ende)
            neu = nachgehoert_annehmen(roh, h, [a.text for a in weg if a.start < l.ende and a.end > l.start])
            worte = sum(len(x["words"]) for x in neu)
            log.info("Nachgehört %.1f bis %.1f s (%s, Fenster %.0f s, Umfeld %.0f s): %d von %d Segmenten, %d Wörter, unsicher %d", l.start, l.ende,
                     l.grund or "ausgelassen", v.fenster_s, v.umfeld_s, len(neu), len(roh), worte, sum(not sicher(x) for x in roh))
            if neu:
                gut = sorted([*gut, *neu], key=lambda s: s["start"])
                dazu.append({"start": round(neu[0]["start"], 3), "end": round(neu[-1]["end"], 3), "grund": l.grund or "ausgelassen", "woerter": worte})
    rest = [l.als_dict() for l in luecken.finden(stellen, gut)]
    for r in rest:
        log.warning("Sprache ohne Text: %.1f bis %.1f s", r["start"], r["end"])
    return gut, dazu, rest


def vollstaendig(motor: Any, pfad: str, sprache: str | None, hinweis: str = "", abschnitt_s: float | None = None) -> dict:
    """Hören, aussortieren, Fehlendes nachholen, zu Abschnitten zusammenfassen."""
    ton = motor.ton(pfad)
    stellen = motor.stellen(ton)
    roh = motor.erkennen(ton, sprache, hinweis, abschnitt_s, stellen)
    gut, weg = aussortieren(roh.get("segments", []), hinweis)
    gut, dazu, rest = nachholen(motor, ton, stellen, gut, weg, sprache or roh.get("language"), hinweis)
    return {"segments": zusammenfassen(gut), "language": roh.get("language"), "abschnitte": roh.get("abschnitte", []),
            "nachgehoert": dazu, "luecken": rest}


def hauptmodell(e: Einstellungen) -> str:
    """Das Modell dieses Dienstes: KLANGWERK_MODELL der Dienstdatei, sonst die Vorgabe."""
    return e.modell or VORGABE


def modelle(e: Einstellungen) -> list[str]:
    """Die Repos, die der Dienst lädt (Modellprüfung ohne Laden, MR-11)."""
    return [hauptmodell(e)]


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = hauptmodell(e)
    name = NAMEN.get(modell, modell)
    motor = motor or Motor(modell)
    rechner = Rechner("hoeren", modell, e)
    app = anwendung("hoeren", rechner, motor.laden, "Hören")

    async def ganz(pfad: Path, sprache: str, hinweis: str = "", abschnitt_s: float | None = None) -> dict:
        gehoert = await rechner.rechnen(vollstaendig, motor, str(pfad), sprachcode(sprache), hinweis, abschnitt_s)
        return {"text": " ".join(s["text"].strip() for s in gehoert["segments"]), "segments": gehoert["segments"], "speakers": [],
                "language": gehoert["language"] or sprache, "model": name, "abschnitte": gehoert["abschnitte"],
                "nachgehoert": gehoert["nachgehoert"], "luecken": gehoert["luecken"]}

    @app.post("/stt")
    async def stt(file: UploadFile = File(...), language: str = Form("german"), diarize: bool = Form(True),
                  prompt: str = Form(""), abschnitt_s: float | None = Form(None)):
        hinweis = hinweis_pruefen(prompt)
        abschnitt_s = abschnitt_pruefen(abschnitt_s)
        pfad = await datei_annehmen(file, e)
        try:
            return await ganz(pfad, language, hinweis, abschnitt_s)
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
            ergebnis = await ganz(pfad, language or "auto", hinweis)
        finally:
            wegraeumen(pfad)
        if response_format == "text":
            return PlainTextResponse(ergebnis["text"])
        if response_format == "json":
            return {"text": ergebnis["text"]}
        return {"text": ergebnis["text"], "language": ergebnis["language"], "segments": ergebnis["segments"], "luecken": ergebnis["luecken"]}

    async def strom(pfad: Path, sprache: str, hinweis: str = "") -> AsyncIterator[dict]:
        """Abschnittsweise erkennen und jedes Segment sofort mit absoluten Zeiten liefern."""
        gesamt = await asyncio.to_thread(dauer_s, str(pfad))
        abschnitte = await asyncio.to_thread(motor.abschnitte, str(pfad))
        alle, fehlt, nummer = [], [], 0
        for a in abschnitte:
            if a["end"] - a["start"] < 0.3:
                continue
            fd, teil = tempfile.mkstemp(suffix=".wav", prefix="klangwerk-")
            os.close(fd)
            try:
                await asyncio.to_thread(ausschneiden, str(pfad), a["start"], a["end"], teil)
                gehoert = await rechner.rechnen(vollstaendig, motor, teil, sprachcode(sprache), hinweis)
                fehlt += [{**l, "start": round(l["start"] + a["start"], 3), "end": round(l["end"] + a["start"], 3)} for l in gehoert["luecken"]]
                for s in gehoert["segments"]:
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
        yield {"_stats": {"model": name, "language": sprache, "total_chunks": nummer, "total_duration": round(gesamt, 1), "luecken": fehlt},
               "_text": " ".join(s["text"].strip() for s in alle)}

    @app.post("/stt/stream")
    async def stt_strom(file: UploadFile = File(...), language: str = Form("german"), prompt: str = Form("")):
        hinweis = hinweis_pruefen(prompt)
        pfad = await datei_annehmen(file, e)

        async def ereignisse():
            try:
                yield f"data: {json.dumps({'type': 'start', 'model': name})}\n\n"
                segmente, beginn, stats = [], time.time(), {}
                async for s in strom(pfad, language, hinweis):
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
