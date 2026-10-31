"""Abschnitte für Hördienste: eine Aufnahme in Teile, die ein Modell am Stück hört, geschnitten nur in Sprechpausen.

Gemeinsam für Whisper (Fenster bis 28 s) und MOSS-Transcribe-Diarize (Fenster bis 60 oder 120 s); jeder Dienst nennt
nur seine Obergrenze. Grundsätze (HELF-86, HELF-87):

- Geschnitten wird nie an fester Zeit, sondern in einer Sprechpause. Die Obergrenze ist nur eine Grenze, kein Takt.
  Gesucht wird die längste Pause im letzten Drittel vor der Grenze (lange Pausen sind eher Satzenden), stufenweise
  ab 500, 300, 200 und 100 ms; gibt es dort keine, die längste Pause im ganzen Abschnitt.
- Gibt es im ganzen Abschnitt keine einzige Pause, wird an der Grenze geschnitten und der Schnitt als Notschnitt
  gekennzeichnet und gemeldet (nie still).
- Ohne Sprachpausenerkennung (Silero) gibt es keine Abschnitte: das ist ein Fehler, kein Rückfall auf feste Teile.
- Ein Abschnitt endet höchstens `polster_s` hinter der letzten Sprache und beginnt höchstens `polster_s` vor der ersten;
  Strecken ohne Sprache fallen weg (MOSS erfindet in Stille Sätze oder lässt Marken fallen, #59 und #61).

Die Pausen kommen aus Silero mit kleinster Pause 100 ms und ohne Polster, also die rohen Lücken zwischen Sprache.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("klangwerk.abschnitte")
RATE = 16000
KLEINSTE_PAUSE_MS = 100
STUFEN_S = (0.5, 0.3, 0.2, 0.1)      # Pausenlängen, die im letzten Drittel nacheinander gesucht werden
POLSTER_S = 0.1


class SprachpausenFehlen(RuntimeError):
    """Ohne Sprachpausenerkennung schneidet kein Hördienst: ein fester Schnitt zerhackt Sätze."""


@dataclass(frozen=True)
class Pause:
    start: float
    ende: float

    @property
    def laenge(self) -> float:
        return self.ende - self.start

    @property
    def mitte(self) -> float:
        return (self.start + self.ende) / 2


@dataclass(frozen=True)
class Abschnitt:
    start: float
    ende: float
    notschnitt: bool = False          # endet ohne Pause an der Obergrenze (gemeldet)

    @property
    def laenge(self) -> float:
        return self.ende - self.start

    def als_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.ende, 3), "notschnitt": self.notschnitt}


def pausen_aus(sprache: Sequence[tuple[float, float]]) -> list[Pause]:
    """Die Lücken zwischen aufeinanderfolgenden Sprachstellen."""
    return [Pause(a[1], b[0]) for a, b in zip(sprache, sprache[1:]) if b[0] > a[1]]


def _schnittstelle(p: Pause, polster_s: float) -> tuple[float, float]:
    """Ende des Abschnitts vor der Pause und Beginn des nächsten danach (in einer langen Pause nicht die ganze Stille)."""
    return min(p.mitte, p.start + polster_s), max(p.mitte, p.ende - polster_s)


def _pause_waehlen(pausen: list[Pause], von: float, grenze: float, hoechstens_s: float, polster_s: float) -> Pause | None:
    kandidaten = [p for p in pausen if p.start > von and _schnittstelle(p, polster_s)[0] <= grenze]
    if not kandidaten:
        return None
    spaet_ab = grenze - hoechstens_s / 3
    for stufe in STUFEN_S:
        spaet = [p for p in kandidaten if p.mitte >= spaet_ab and p.laenge >= stufe - 1e-9]
        if spaet:
            return max(spaet, key=lambda p: (round(p.laenge, 2), p.mitte))
    return max(kandidaten, key=lambda p: (round(p.laenge, 2), p.mitte))


def planen(sprache: Sequence[tuple[float, float]], gesamt_s: float, hoechstens_s: float, *,
           polster_s: float = POLSTER_S) -> list[Abschnitt]:
    """Abschnitte von höchstens `hoechstens_s` über die Sprachstellen `sprache` (Sekunden, aufsteigend). Ohne Sprache
    keine Abschnitte."""
    if hoechstens_s <= 2 * polster_s:
        raise ValueError(f"Obergrenze {hoechstens_s} s zu klein")
    sprache = [(max(0.0, a), min(gesamt_s, b)) for a, b in sprache if b > a]
    if not sprache:
        return []
    pausen = pausen_aus(sprache)
    von = max(0.0, sprache[0][0] - polster_s)
    schluss = min(gesamt_s, sprache[-1][1] + polster_s)
    aus: list[Abschnitt] = []
    while schluss - von > hoechstens_s:
        grenze = von + hoechstens_s
        p = _pause_waehlen(pausen, von, grenze, hoechstens_s, polster_s)
        if p is None:
            log.warning("Notschnitt bei %.1f s: keine Sprechpause zwischen %.1f und %.1f s", grenze, von, grenze)
            aus.append(Abschnitt(von, grenze, notschnitt=True))
            von = grenze
            continue
        ende, weiter = _schnittstelle(p, polster_s)
        aus.append(Abschnitt(von, ende))
        von = weiter
    aus.append(Abschnitt(von, schluss))
    return aus


def sprachbeginn_vor(sprache: Sequence[tuple[float, float]], zeit: float, polster_s: float = POLSTER_S,
                     spielraum_s: float = 1.0) -> float:
    """Beginn für einen Vorlauf ab `zeit` (Beginn eines Beitrags laut Modell): der Beginn der Sprachstelle davor, wenn
    er höchstens `spielraum_s` früher liegt (so beginnt der Vorlauf in einer Pause statt mitten im Wort), sonst `zeit`;
    jeweils mit Polster."""
    beginn = zeit
    for a, _ in sprache:
        if a > zeit:
            break
        beginn = a
    if zeit - beginn > spielraum_s:
        beginn = zeit
    return max(0.0, beginn - polster_s)


class Sprachpausen:
    """Silero als Sprachpausenerkennung (Prozessor, klein)."""

    def __init__(self) -> None:
        self.vad: Any = None

    def laden(self) -> None:
        from silero_vad import load_silero_vad
        self.vad = load_silero_vad()

    def stellen(self, ton: np.ndarray) -> list[tuple[float, float]]:
        """Sprachstellen eines Tons (16 kHz mono) in Sekunden."""
        if self.vad is None:
            raise SprachpausenFehlen("keine Sprachpausenerkennung geladen (silero-vad)")
        import torch
        from silero_vad import get_speech_timestamps
        roh = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(ton, dtype=np.float32)), self.vad,
                                    sampling_rate=RATE, min_silence_duration_ms=KLEINSTE_PAUSE_MS, speech_pad_ms=0)
        return [(s["start"] / RATE, s["end"] / RATE) for s in roh]

    def abschnitte(self, ton: np.ndarray, hoechstens_s: float) -> list[Abschnitt]:
        return planen(self.stellen(ton), len(ton) / RATE, hoechstens_s)
