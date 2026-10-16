"""Gemeinsames der Sprecher-Dienste: Audio einmal nach 16 kHz mono wandeln (ffmpeg)."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

RATE = 16000


def mono_16k(pfad: str) -> np.ndarray:
    fd, ziel = tempfile.mkstemp(suffix=".wav", prefix="klangwerk-")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", pfad, "-ar", str(RATE), "-ac", "1", ziel], check=True)
        daten, _ = sf.read(ziel, dtype="float32", always_2d=False)
        return daten
    finally:
        os.unlink(ziel)
