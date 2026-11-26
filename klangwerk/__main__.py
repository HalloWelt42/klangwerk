"""Startpunkt: `klangwerk <dienst> [--port N]`. Je Dienst ein eigener Prozess mit genau einem Modell (der Tondienst ohne)."""

from __future__ import annotations

import argparse
import importlib
import sys

from .unterbau import Einstellungen, starten

DIENSTE = {
    "sprechen": "Sprachsynthese (Qwen3-TTS)",
    "musik": "Musik aus Beschreibung und Liedtext (MiniMax-Music3 in MLX, Umgebung .venv-moss)",
    "klonen": "Stimmklon (Qwen3-TTS Base)",
    "hoeren": "Spracherkennung (Whisper Large V3, als Variante 8 Bit oder Turbo über KLANGWERK_MODELL)",
    "hoeren-moss": "Spracherkennung mit Sprechern (MOSS-Transcribe-Diarize über MLX, eigene Umgebung .venv-moss)",
    "sprecher": "Stimmabdruck je Abschnitt (ECAPA)",
    "trennen": "Wer spricht wann (pyannote)",
    "ton": "Tonarbeit der Apps (ffmpeg, ohne Modell)",
}


def main() -> None:
    p = argparse.ArgumentParser(prog="klangwerk", description="Klangwerk: Sprachdienste, je Prozess ein Modell")
    p.add_argument("dienst", choices=sorted(DIENSTE), help="; ".join(f"{k}: {v}" for k, v in DIENSTE.items()))
    p.add_argument("--port", type=int, help="Vorgabe je Dienst (8781 bis 8787)")
    p.add_argument("--pruefen", action="store_true",
                   help="nur prüfen, ob die Modelle des Dienstes im Zwischenspeicher liegen, ohne zu laden und ohne Netz; "
                        "endet mit 0 (vorhanden) oder 1 (fehlt), die letzte Zeile nennt den Grund")
    a = p.parse_args()
    modul = importlib.import_module(f"klangwerk.{a.dienst.replace('-', '_')}")
    einstellungen = Einstellungen()
    if a.pruefen:
        sys.exit(pruefen(modul.modelle(einstellungen)))
    starten(modul.erstellen(einstellungen), a.port or modul.PORT, einstellungen)


def pruefen(repos: list[str]) -> int:
    """Modellprüfung für den Maschinenraum (Feld modellpruefung der Dienstdatei)."""
    from .modellpruefung import fehlt
    grund = fehlt(repos)
    print(f"fehlt: {grund}" if grund else f"vorhanden: {', '.join(repos) or 'kein Modell nötig'}")
    return 1 if grund else 0


if __name__ == "__main__":
    main()
