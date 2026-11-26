"""Ob die Modelle eines Dienstes im Zwischenspeicher von Hugging Face liegen, ohne sie zu laden (MR-11).

`klangwerk <dienst> --pruefen` endet mit 0, wenn alles da ist, sonst mit 1; die letzte Zeile der Ausgabe nennt den Grund.
So erwartet es die Modellprüfung des Maschinenraums (Feld `modellpruefung` der Dienstdatei). Geprüft wird ohne Netz:
Der Schnappschuss jedes Repos liegt im Zwischenspeicher (`HF_HUB_CACHE`), und jede seiner Dateien lässt sich öffnen.
Zeigt ein Link ins Leere, etwa weil das Modell mit Links aufs Hauptlager ausgelagert wurde und das Hauptlager nicht
eingehängt ist, sagt der Grund das.
"""

from __future__ import annotations

import os
from pathlib import Path


class ModellFehlt(RuntimeError):
    """Ein Repo liegt nicht vollständig im Zwischenspeicher."""


def schnappschuss(repo: str) -> Path:
    """Ordner des Schnappschusses im Zwischenspeicher, ohne Netz; ModellFehlt, wenn es ihn nicht gibt."""
    from huggingface_hub import constants, snapshot_download
    try:
        return Path(snapshot_download(repo, local_files_only=True))
    except Exception as f:                       # LocalEntryNotFoundError, IncompleteSnapshotError, je nach Fassung
        raise ModellFehlt(f"{repo} fehlt im Zwischenspeicher {constants.HF_HUB_CACHE} ({type(f).__name__})") from f


def pruefen(repo: str) -> Path:
    """Schnappschuss, dessen Dateien sich alle öffnen lassen; sonst ModellFehlt mit Grund."""
    ordner = schnappschuss(repo)
    leer = [p for p in sorted(ordner.rglob("*")) if p.is_symlink() and not p.exists()]
    if leer:
        raise ModellFehlt(f"{repo}: {len(leer)} Links ins Leere, etwa {os.path.realpath(leer[0])} (Hauptlager nicht eingehängt?)")
    if not any(p.is_file() for p in ordner.rglob("*")):
        raise ModellFehlt(f"{repo}: Schnappschuss ohne Dateien in {ordner}")
    return ordner


def fehlt(repos: list[str]) -> str | None:
    """Grund für das erste Repo, das nicht vollständig da ist; None, wenn alle da sind."""
    for repo in repos:
        try:
            pruefen(repo)
        except ModellFehlt as f:
            return str(f)
    return None
