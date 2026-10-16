"""Klangwerk: Sprachdienste je Modell. Jeder Dienst ist ein eigener Prozess mit genau einem Modell.

Die Fassung steht in `version.json` im Wurzelordner; eine Installation legt sie als `VERSION` ins Paket.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fassung() -> str:
    hier = Path(__file__).resolve().parent
    if (hier / "VERSION").is_file():
        return (hier / "VERSION").read_text(encoding="utf-8").strip()
    quelle = hier.parent / "version.json"
    if quelle.is_file():
        return json.loads(quelle.read_text(encoding="utf-8"))["version"]
    return "0.0.0"


__version__ = _fassung()
