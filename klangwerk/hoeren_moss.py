"""Dienst hoeren-moss: Spracherkennung mit Sprechertrennung, MOSS-Transcribe-Diarize 0.9B über MLX (mlx-audio).

Das Modell (Whisper-Medium als Hörteil, Qwen3-0.6B als Textteil) schreibt Ton als Folge `[Beginn][S01]Text[Ende]`:
Zeiten in Sekunden und eine Sprecherkennung je Beitrag. Wortzeiten gibt es nicht, nur Beitragszeiten.

Abschnitte (HELF-86, HELF-87): Das Modell hört nie eine ganze lange Aufnahme am Stück. Am 31.10.2026 wuchs es dabei
(torch auf der Grafik, float32, 7 min am Stück) auf über 100 GB und brachte den Mac zum Absturz. Gehört wird in
Abschnitten bis zur Obergrenze (`KLANGWERK_ABSCHNITT_S`, Vorgabe 60 s, je Auftrag `abschnitt_s` bis höchstens
`KLANGWERK_ABSCHNITT_HOECHSTENS_S`, der sicheren Länge aus der Messreihe), geschnitten nur in
Sprechpausen (gemeinsames Modul `abschnitte`, wie beim Dienst hoeren). Je Abschnitt ein eigener Durchgang, danach wird
der Zwischenspeicher von MLX geleert; die Speichergrenze des Maschinenraums gilt auch im Prozess (unterbau).

Zusammenführen nach Zeit: Jeder Abschnitt ab dem zweiten beginnt mit dem letzten Beitrag des vorigen als Vorlauf
(höchstens `VORLAUF_HOECHSTENS_S`, Beginn in einer Sprechpause). Aus einem Abschnitt zählen nur Beiträge, die nach
seinem Schnitt beginnen; der Schnitt liegt in einer Pause, also reicht kein Beitrag über ihn. Den Vorlauf hört das
Modell doppelt: seine Kennung im neuen Abschnitt ist die des Beitrags im alten (erstes Zeichen für die Sprecher).

Sprecher über Abschnitte: Die Kennungen des Modells gelten nur innerhalb eines Abschnitts. Zugeordnet wird je
Abschnitt und Kennung, nicht je Beitrag: zuerst über den Vorlauf, sonst über den Stimmabdruck (ECAPA, Mittel über die
Beiträge ab 1 s, Kosinus gegen das Mittel jedes bisherigen Sprechers, Schwelle `KLANGWERK_SPRECHER_SCHWELLE`). Zwei
Kennungen eines Abschnitts werden nie derselbe Sprecher. Was keiner passt, wird ein neuer Sprecher. Wie sicher die
Zuordnung ist, steht je Segment in `speaker_sicher`.

Formatfehler (keine Marken, chinesische Zeichen bei anderer Sprache): der Abschnitt wird einmal mit strengerer Anweisung
wiederholt; bleibt der Fehler, kommt sein Text ohne Sprecher und mit Warnung (`warnungen`), nie still.

Schnittstelle wie der Dienst hoeren (`POST /stt`, `POST /v1/audio/transcriptions`); dazu `abschnitte` (Plan und
Messwerte je Abschnitt) und `warnungen`.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .abschnitte import RATE, Abschnitt, Sprachpausen, planen, sprachbeginn_vor
from .hoeren import abschnitt_pruefen, hinweis_pruefen, sprachcode
from .unterbau import Einstellungen, Rechner, anwendung, datei_annehmen, wegraeumen

log = logging.getLogger("klangwerk.hoeren_moss")
PORT = 8787
VORGABE = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
NAMEN = {"OpenMOSS-Team/MOSS-Transcribe-Diarize": "MOSS-Transcribe-Diarize 0.9B"}
ABSCHNITT_VORGABE_S = 60             # Start laut HELF-86; Ziel 120 s nach der Messreihe (Dienstdatei)
ABSCHNITT_OBERGRENZE_S = 240         # mehr nimmt der Dienst je Auftrag nicht an (ohne KLANGWERK_ABSCHNITT_HOECHSTENS_S)
VORLAUF_HOECHSTENS_S = 10.0
NAHT_S = 0.5                         # Spielraum der Zeitmarken des Modells am Schnitt
TOKEN_JE_S = 12                      # Obergrenze der erzeugten Zeichen je Sekunde Ton (gesprochen sind es 4 bis 8)
TOKEN_ZUSCHLAG = 64
MERKMAL_MINDESTENS_S = 1.0           # darunter trägt ein Beitrag keinen brauchbaren Stimmabdruck (#60)
MERKMAL_BEITRAEGE = 6                # je Kennung und Abschnitt höchstens so viele (die längsten)
SICHER_ABSTAND = 0.1                 # über Schwelle plus Abstand gilt eine Zuordnung über die Stimme als sicher
SEGMENT_RE = re.compile(r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]", re.DOTALL)
CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
CJK_SPRACHEN = {"zh", "ja", "ko", "yue"}
STRENG = (" Write the transcript verbatim in the language that is spoken; never translate. Every segment must have "
          "the form [start][Sxx]text[end].")


def anweisung(hinweis: str, streng: bool = False) -> str:
    from mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize import DEFAULT_PROMPT   # eine Wahrheit
    text = DEFAULT_PROMPT + (STRENG if streng else "")
    return f"{text} Hotwords: {hinweis}" if hinweis else text


def hoechstens_token(dauer: float) -> int:
    return math.ceil(dauer * TOKEN_JE_S) + TOKEN_ZUSCHLAG


def _vergleich(text: str) -> str:
    return " ".join("".join(z for z in text.lower() if z.isalnum() or z.isspace()).split())


@dataclass(frozen=True)
class Beitrag:
    start: float
    ende: float
    sprecher: str | None
    text: str
    sicher: bool = True


def beitraege_lesen(text: str, versatz: float = 0.0) -> list[Beitrag]:
    """Ausgabe des Modells zu Beiträgen, Zeiten um `versatz` (Beginn des gehörten Stücks) verschoben."""
    aus = []
    for m in SEGMENT_RE.finditer(text):
        a, b = float(m.group("start")), float(m.group("end"))
        if (t := m.group("text").strip()) and b >= a:
            aus.append(Beitrag(a + versatz, b + versatz, m.group("speaker"), t))
    return aus


def formatfehler(text: str, beitraege: list[Beitrag], sprache: str | None) -> str | None:
    if text.strip() and not beitraege:
        return "keine Marken [Sxx]"
    if sprache and sprache not in CJK_SPRACHEN and CJK_RE.search(text):
        return "chinesische Zeichen"
    return None


def segmente(beitraege: list[Beitrag], hinweis: str = "") -> list[dict]:
    """Beiträge im Format des Dienstes hoeren: leere, nur wiederholte und reine Hinweis-Beiträge fallen weg, Zeiten
    aufsteigend, Ende nie vor dem Beginn."""
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
        aus.append({"start": start, "end": round(max(b.ende, start), 3), "text": text, "words": [], "speaker": b.sprecher,
                    "speaker_sicher": b.sicher})
    return aus


def sprecherliste(segs: list[dict]) -> list[str]:
    return sorted({s["speaker"] for s in segs if s.get("speaker")})


# ── Sprecher über Abschnitte ─────────────────────────────────────────────────

@dataclass
class Zuordnung:
    sprecher: str
    sicher: bool
    wie: str                          # vorlauf, stimme, neu


class Sprecherbuch:
    """Sprecher der ganzen Aufnahme: je Sprecher die Summe seiner Stimmabdrücke (normiert verglichen)."""

    def __init__(self, schwelle: float) -> None:
        self.schwelle = schwelle
        self.summen: dict[str, np.ndarray] = {}
        self.zahl = 0

    def neu(self) -> str:
        self.zahl += 1
        return f"S{self.zahl:02d}"

    def aehnlichkeit(self, merkmal: np.ndarray, sprecher: str) -> float:
        s = self.summen.get(sprecher)
        if s is None:
            return -1.0
        n = float(np.linalg.norm(s)) * float(np.linalg.norm(merkmal))
        return float(np.dot(s, merkmal) / n) if n else -1.0

    def zuordnen(self, merkmale: dict[str, np.ndarray | None], fest: dict[str, str]) -> dict[str, Zuordnung]:
        """Kennungen eines Abschnitts auf Sprecher: `fest` (aus dem Vorlauf) zuerst, dann die ähnlichsten Paare über der
        Schwelle, der Rest neu. Zwei Kennungen eines Abschnitts nie auf denselben Sprecher."""
        aus = {k: Zuordnung(s, True, "vorlauf") for k, s in fest.items()}
        besetzt = set(fest.values())
        paare = sorted(((self.aehnlichkeit(m, s), k, s) for k, m in merkmale.items() if k not in aus and m is not None
                        for s in self.summen), reverse=True)
        for wert, k, s in paare:
            if wert < self.schwelle:
                break
            if k in aus or s in besetzt:
                continue
            aus[k] = Zuordnung(s, wert >= self.schwelle + SICHER_ABSTAND, "stimme")
            besetzt.add(s)
        for k in merkmale:
            if k not in aus:
                aus[k] = Zuordnung(self.neu(), merkmale[k] is not None, "neu")
        for k, m in merkmale.items():
            if m is not None:
                s = aus[k].sprecher
                self.summen[s] = self.summen.get(s, np.zeros_like(m)) + m / (float(np.linalg.norm(m)) or 1.0)
        return aus


def kennungsmerkmale(beitraege: list[Beitrag], abdruecke: Callable[[list[dict]], list[list[float] | None]]
                     ) -> dict[str, np.ndarray | None]:
    """Je Kennung eines Abschnitts ein Stimmabdruck: Mittel über ihre längsten Beiträge ab 1 s; ohne solche None."""
    je: dict[str, list[Beitrag]] = {}
    for b in beitraege:
        if b.sprecher:
            je.setdefault(b.sprecher, []).append(b)
    aus: dict[str, np.ndarray | None] = {}
    for k, liste in je.items():
        lang = sorted((b for b in liste if b.ende - b.start >= MERKMAL_MINDESTENS_S), key=lambda b: b.start - b.ende)
        roh = abdruecke([{"start": b.start, "end": b.ende} for b in lang[:MERKMAL_BEITRAEGE]]) if lang else []
        vektoren = [np.asarray(v, dtype=np.float32) for v in roh if v is not None]
        aus[k] = np.mean(vektoren, axis=0) if vektoren else None
    return aus


@dataclass
class Naht:
    """Ergebnis eines Abschnitts nach dem Zusammenführen an seinem Schnitt."""
    neu: list[Beitrag]
    doppelt: list[Beitrag]
    ersetzt_vorige: int = 0           # Beiträge des vorigen Abschnitts, die ein über den Schnitt gezogener ersetzt


def naht(beitraege: list[Beitrag], schnitt: float) -> Naht:
    """Teilt die Beiträge eines Abschnitts mit Vorlauf: vor dem Schnitt beginnende sind doppelt gehört (Vorlauf), nur
    die danach zählen. Hat das Modell den Vorlauf mit dem ersten neuen Beitrag zusammengezogen (er reicht deutlich über
    den Schnitt), zählt er ganz und ersetzt den Vorlauf im vorigen Abschnitt."""
    neu, doppelt, ersetzt = [], [], 0
    for b in beitraege:
        if b.start >= schnitt - NAHT_S:
            neu.append(b)
        elif b.ende > schnitt + 2 * NAHT_S:
            neu.append(b)
            ersetzt += 1
        else:
            doppelt.append(b)
    return Naht(neu, doppelt, ersetzt)


def vorlauf_kennung(doppelt: list[Beitrag], vorlauf: Beitrag) -> str | None:
    """Die Kennung, unter der das Modell den Vorlauf im neuen Abschnitt gehört hat (größte Überlappung)."""
    best, beste = None, 0.0
    for b in doppelt:
        ueber = min(b.ende, vorlauf.ende) - max(b.start, vorlauf.start)
        if b.sprecher and ueber > beste:
            best, beste = b.sprecher, ueber
    return best


# ── Modell ───────────────────────────────────────────────────────────────────

@dataclass
class Gehoert:
    text: str
    prompt_token: int
    erzeugt_token: int
    sekunden: float
    spitze_gb: float


@dataclass
class Bericht:
    beitraege: list[Beitrag]
    abschnitte: list[dict] = field(default_factory=list)
    warnungen: list[str] = field(default_factory=list)


class Motor:
    """Das Modell in MLX, dazu Silero (Sprechpausen) und ECAPA (Stimmabdruck) auf dem Prozessor."""

    def __init__(self, modell: str, genauigkeit: str, abdruck: Any = None) -> None:
        self.modell, self.genauigkeit = modell, genauigkeit
        self.netz: Any = None
        self.pausen = Sprachpausen()
        if abdruck is None:
            from .sprecher import VORGABE as ECAPA, Motor as Abdruck
            abdruck = Abdruck(ECAPA)
        self.abdruck = abdruck

    def laden(self) -> bool:
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model
        netz = load_model(self.modell)
        netz.set_dtype(mx.bfloat16 if self.genauigkeit == "bfloat16" else mx.float32)
        mx.eval(netz.parameters())
        self.netz = netz
        self.pausen.laden()
        self.abdruck.laden()
        log.info("MOSS-Transcribe-Diarize über MLX in %s geladen", self.genauigkeit)
        return True

    def hoeren(self, stueck: np.ndarray, anweisung_text: str) -> Gehoert:
        import mlx.core as mx
        mx.reset_peak_memory()
        beginn = time.monotonic()
        aus = self.netz.generate(stueck, max_tokens=hoechstens_token(len(stueck) / RATE), temperature=0.0,
                                 prompt=anweisung_text)
        gehoert = Gehoert(aus.text, int(aus.prompt_tokens), int(aus.generation_tokens), time.monotonic() - beginn,
                          mx.get_peak_memory() / 1024**3)
        mx.clear_cache()
        return gehoert

    def erkennen(self, pfad: str, hinweis: str = "", sprache: str | None = None, abschnitt_s: float = ABSCHNITT_VORGABE_S,
                 schwelle: float = 0.5) -> Bericht:
        from .stimmdaten import mono_16k
        ton = mono_16k(pfad)
        sprache_stellen = self.pausen.stellen(ton)
        plan = planen(sprache_stellen, len(ton) / RATE, abschnitt_s)
        return zusammenfuehren(ton, plan, sprache_stellen, lambda s, streng: self.hoeren(s, anweisung(hinweis, streng)),
                               lambda zeiten: self.abdruck.abdruecke_ton(ton, zeiten), sprache, Sprecherbuch(schwelle))


def zusammenfuehren(ton: np.ndarray, plan: list[Abschnitt], sprache_stellen: list[tuple[float, float]],
                    hoeren: Callable[[np.ndarray, bool], Gehoert],
                    abdruecke: Callable[[list[dict]], list[list[float] | None]], sprache: str | None,
                    buch: Sprecherbuch) -> Bericht:
    """Hört Abschnitt für Abschnitt (mit Vorlauf) und führt nach Zeit und Sprechern zusammen (siehe Modultext)."""
    bericht = Bericht([])
    vorher: Abschnitt | None = None
    for nr, a in enumerate(plan, 1):
        letzter = bericht.beitraege[-1] if bericht.beitraege else None
        vorlauf = (letzter if vorher is not None and letzter is not None and letzter.start >= vorher.start
                   and a.start - letzter.start <= VORLAUF_HOECHSTENS_S and letzter.sprecher else None)
        beginn = max(vorher.start if vorher else 0.0, sprachbeginn_vor(sprache_stellen, vorlauf.start)) if vorlauf else a.start
        stueck = ton[int(beginn * RATE):int(a.ende * RATE)]
        g = hoeren(stueck, False)
        roh = beitraege_lesen(g.text, beginn)
        fehler, wiederholt = formatfehler(g.text, roh, sprache), False
        if fehler:
            log.warning("Abschnitt %d (%.1f bis %.1f s): %s, wird mit strengerer Anweisung wiederholt", nr, a.start, a.ende, fehler)
            g2 = hoeren(stueck, True)
            roh2 = beitraege_lesen(g2.text, beginn)
            fehler2 = formatfehler(g2.text, roh2, sprache)
            wiederholt = True
            g = Gehoert(g2.text, g2.prompt_token, g.erzeugt_token + g2.erzeugt_token, g.sekunden + g2.sekunden,
                        max(g.spitze_gb, g2.spitze_gb))
            roh, fehler = roh2, fehler2
        if fehler:
            satz = f"Abschnitt {nr} ({a.start:.1f} bis {a.ende:.1f} s): {fehler} auch nach Wiederholung, Text ohne Sprecher"
            bericht.warnungen.append(satz)
            log.warning(satz)
            roh = [Beitrag(beginn, a.ende, None, g.text.strip(), sicher=False)] if g.text.strip() else []
        n = naht(roh, a.start) if vorlauf else Naht(roh, [])
        if n.ersetzt_vorige:
            bericht.beitraege = [b for b in bericht.beitraege if b.start < min(x.start for x in n.neu) - NAHT_S]
        fest = {}
        if vorlauf and (k := vorlauf_kennung(n.doppelt, vorlauf)):
            fest[k] = vorlauf.sprecher
        merkmale = kennungsmerkmale(n.neu, abdruecke)
        zuordnung = buch.zuordnen(merkmale, {k: s for k, s in fest.items() if k in merkmale})
        for b in n.neu:
            z = zuordnung.get(b.sprecher) if b.sprecher else None
            bericht.beitraege.append(replace(b, sprecher=z.sprecher if z else None,
                                             sicher=bool(z and z.sicher and b.sicher)))
        bericht.abschnitte.append({**a.als_dict(), "nummer": nr, "gehoert_ab": round(beginn, 3),
                                   "vorlauf": vorlauf is not None, "sekunden": round(g.sekunden, 2),
                                   "prompt_token": g.prompt_token, "erzeugt_token": g.erzeugt_token,
                                   "spitze_gb": round(g.spitze_gb, 2), "wiederholt": wiederholt, "fehler": fehler,
                                   "naht_zusammengezogen": n.ersetzt_vorige,
                                   "sprecher": {k: {"als": z.sprecher, "wie": z.wie, "sicher": z.sicher}
                                                for k, z in zuordnung.items()}})
        log.info("Abschnitt %d/%d %.1f bis %.1f s (gehört ab %.1f): %.1f s gerechnet, %d Zeichen, Spitze MLX %.2f GB",
                 nr, len(plan), a.start, a.ende, beginn, g.sekunden, g.erzeugt_token, g.spitze_gb)
        vorher = a
    for a in plan:
        if a.notschnitt:
            bericht.warnungen.append(f"Notschnitt ohne Sprechpause bei {a.ende:.1f} s")
    return bericht


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = e.modell or VORGABE
    name = NAMEN.get(modell, modell)
    hoechstens_s = e.abschnitt_hoechstens_s or ABSCHNITT_OBERGRENZE_S
    vorgabe_s = min(e.abschnitt_s or ABSCHNITT_VORGABE_S, hoechstens_s)
    motor = motor or Motor(modell, e.genauigkeit)
    rechner = Rechner("hoeren-moss", modell, e)
    app = anwendung("hoeren-moss", rechner, motor.laden, "Hören (MOSS)")

    async def ganz(pfad: Path, sprache: str | None, hinweis: str = "", abschnitt_s: float | None = None) -> dict:
        code = sprachcode(sprache)
        bericht = await rechner.rechnen(motor.erkennen, str(pfad), hinweis, code, abschnitt_s or vorgabe_s,
                                        e.sprecher_schwelle)
        segs = segmente(bericht.beitraege, hinweis)
        return {"text": " ".join(s["text"] for s in segs), "segments": segs, "speakers": sprecherliste(segs),
                "language": code, "model": name, "abschnitte": bericht.abschnitte, "warnungen": bericht.warnungen}

    @app.post("/stt")
    async def stt(file: UploadFile = File(...), language: str = Form("german"), diarize: bool = Form(True),
                  prompt: str = Form(""), abschnitt_s: float | None = Form(None)):
        hinweis = hinweis_pruefen(prompt)
        abschnitt_s = abschnitt_pruefen(abschnitt_s, hoechstens_s)
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
            ergebnis = await ganz(pfad, language, hinweis)
        finally:
            wegraeumen(pfad)
        if response_format == "text":
            return PlainTextResponse(ergebnis["text"])
        if response_format == "json":
            return {"text": ergebnis["text"]}
        return {"text": ergebnis["text"], "language": ergebnis["language"], "segments": ergebnis["segments"]}

    return app
