"""Selbstauskunft eines Dienstes in seiner Dienstdatei des Maschinenraums: die Sprachen, die sein Modell kann (ISO 639-1),
gehen von dort als Steckbrief an die Leitstelle; Apps richten ihre Sprachwahl danach. Quelle ist das Modell selbst
(beim Hören der Tokenizer von Whisper), keine Liste von Hand.

    python -m klangwerk.selbstauskunft hoeren ~/Maschinenraum/dienste/klangwerk-hoeren.toml

Trägt `sprachen = [...]` hinter `faehigkeiten` ein, wenn die Datei noch keine Sprachen nennt; sonst bleibt sie.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def sprachen(dienst: str) -> list[str]:
    if dienst == "hoeren":
        from mlx_whisper.tokenizer import LANGUAGES
        return [code for code in LANGUAGES if len(code) == 2]      # ohne die dreistelligen (haw, yue)
    raise SystemExit(f"Für {dienst} sind keine Sprachen bekannt")


def eintragen(datei: Path, liste: list[str]) -> bool:
    zeilen = datei.read_text().splitlines(keepends=True)
    if any(z.startswith("sprachen") for z in zeilen):
        return False
    stelle = next(i for i, z in enumerate(zeilen) if z.startswith("faehigkeiten")) + 1
    zeilen.insert(stelle, f"sprachen = {json.dumps(liste)}\n")
    datei.write_text("".join(zeilen))
    return True


if __name__ == "__main__":
    dienst, datei = sys.argv[1], Path(sys.argv[2]).expanduser()
    print(f"Sprachen eingetragen: {datei}" if eintragen(datei, sprachen(dienst)) else f"bleibt: {datei}")
