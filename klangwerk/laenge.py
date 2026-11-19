"""Höchstlänge einer Sprachausgabe nach ihrem Text (Messung 18.10.2026, Vertonung K3: ein Klon lief bis zur Vorgabe
von 4096 Token durch, etwa 340 s Ton und 17 Minuten Rechnung, und wurde danach als Echo verworfen). Bemessen
großzügig: 5 Zeichen je Sekunde (deutsche Sprache liegt bei 12 bis 16), dazu ein fester Rand. Das Modell erzeugt immer
im ursprünglichen Tempo; erst die fertige Ausgabe wird gestaucht. Ihre Tokenbegrenzung bleibt deshalb tempounabhängig.
Was länger würde, ist ein Durchgehen des Modells, kein Text."""

from __future__ import annotations

import math

TOKEN_JE_S = 12            # die Modelle rechnen mit 12 Tonschritten je Sekunde
S_JE_ZEICHEN = 0.2         # 5 Zeichen je Sekunde
RAND_S = 3.0
HOECHSTENS_TOKEN = 4096    # Vorgabe des Modells, nie mehr


def hoechstens_token(text: str) -> int:
    sekunden = RAND_S + S_JE_ZEICHEN * len(text.strip())
    return min(HOECHSTENS_TOKEN, math.ceil(sekunden * TOKEN_JE_S))
