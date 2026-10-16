"""Startpunkt: `klangwerk <dienst> [--port N]`. Je Dienst ein eigener Prozess mit genau einem Modell."""

from __future__ import annotations

import argparse
import importlib

from .unterbau import Einstellungen, starten

DIENSTE = {
    "sprechen": "Sprachsynthese (Qwen3-TTS)",
    "klonen": "Stimmklon (Qwen3-TTS Base)",
    "hoeren": "Spracherkennung (Whisper Large V3)",
    "sprecher": "Stimmabdruck je Abschnitt (ECAPA)",
    "trennen": "Wer spricht wann (pyannote)",
}


def main() -> None:
    p = argparse.ArgumentParser(prog="klangwerk", description="Klangwerk: Sprachdienste, je Prozess ein Modell")
    p.add_argument("dienst", choices=sorted(DIENSTE), help="; ".join(f"{k}: {v}" for k, v in DIENSTE.items()))
    p.add_argument("--port", type=int, help="Vorgabe je Dienst (8781 bis 8785)")
    a = p.parse_args()
    modul = importlib.import_module(f"klangwerk.{a.dienst}")
    einstellungen = Einstellungen()
    starten(modul.erstellen(einstellungen), a.port or modul.PORT, einstellungen)


if __name__ == "__main__":
    main()
