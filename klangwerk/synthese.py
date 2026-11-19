"""Übergang zur Modellbibliothek und gemeinsame Nachbearbeitung der Sprachausgabe."""

from __future__ import annotations

from typing import Any

import numpy as np


def modell_audio(modell: Any, **parameter: Any) -> np.ndarray | None:
    """Die öffentliche Sprachwahl abbilden und sämtliche erzeugten Audioteile übernehmen.

    Die Bibliothek erwartet lang_code; ein unbekanntes language würde sie still in **kwargs annehmen.
    Das Tempo setzt anschließend die Tonverarbeitung um, nicht das Modell.
    """
    parameter["lang_code"] = parameter.pop("language", "auto")
    parameter.pop("speed", None)
    teile = []
    for ergebnis in modell.generate(**parameter):
        if ergebnis.audio is not None:
            audio = np.asarray(ergebnis.audio)
            if audio.size:
                teile.append(audio)
    return np.concatenate(teile) if teile else None


async def mit_tempo(audio: np.ndarray, rate: int, tempo: float) -> np.ndarray:
    """Tempo bei gleicher Tonhöhe mit der gemeinsamen Tonverarbeitung ändern, genau einmal nach der Synthese."""
    if tempo == 1.0:
        return audio
    from appkern.ton import Ton

    ton = Ton()
    # Die öffentliche Schnittstelle erlaubt auch 0,1; die gemeinsame Operation mindestens 0,5 je Schritt.
    while tempo < 0.5:
        audio = await ton.stauchen(audio, rate, 0.5)
        tempo /= 0.5
    return await ton.stauchen(audio, rate, tempo)
