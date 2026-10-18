"""Dienst ton: alle Tonarbeit der Apps auf dem Mac (wandeln, messen, blenden, legen); der Pi liefert nur Daten (Nutzer
18.10.2026: "pi ist nur datenlieferant"). Ohne Modell: bereit heißt, ffmpeg und ffprobe sind da.

Protokoll, Routen und Rechnung kommen aus dem Appkern (`appkern.ton`: `TonHier`, `gegenseite.routen`), damit es jede
Tonoperation genau einmal gibt; die Apps sprechen ihn über die Leitstelle mit `appkern.ton.TonDienst` an. Mehrere
Aufträge laufen nebeneinander (je einer ein eigener ffmpeg-Prozess); erzeugte Dateien liegen bis zum Abholen im
Ergebnisordner, höchstens eine Stunde.
"""

from __future__ import annotations

from .unterbau import Einstellungen, Rechner, anwendung

PORT = 8786


def erstellen(einstellungen: Einstellungen | None = None):
    from appkern.ton import Ton, TonHier, programm
    from appkern.ton.gegenseite import routen

    e = einstellungen or Einstellungen()
    rechner = Rechner("ton", "ffmpeg", e)

    def lader() -> str:
        return f"{programm('ffmpeg')} und {programm('ffprobe')}"

    app = anwendung("ton", rechner, lader, "Ton")
    app.include_router(routen(TonHier(Ton()), e.ergebnisse / "ton"))
    return app
