"""Was gesprochen wurde, muss auch dastehen (BT-91): welche Sprache deckt die Abschrift nicht?

Verglichen werden die Sprachstellen der Aufnahme (Silero) mit den Wörtern der Abschrift, nicht mit den Segmenten: ein
Modell kann ein Segment über zwanzig Sekunden ziehen und darin nur drei Wörter nennen. Eine Lücke ist Sprache ohne Wort
in der Nähe. Was ein Filter aussortiert hat, deckt nichts; liegt dort Sprache, genügt schon wenig davon.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Strecke = tuple[float, float]
WORT_POLSTER_S = 0.35             # Wortzeiten sind nicht genauer
WORT_HOECHSTENS_S = 1.5           # ein längeres "Wort" steht für eine verrutschte Zeit und deckt nur so viel
LUECKE_AB_S = 1.0                 # so viel Sprache muss fehlen; darunter sind es Laute, die niemand schreibt
AUSSORTIERT_AB_S = 0.3            # wo ein Filter etwas strich


@dataclass(frozen=True)
class Luecke:
    start: float                  # hier darf nachgehört werden, ohne ein schon gehörtes Wort zu berühren
    ende: float
    sprache_s: float
    grund: str = ""               # warum ein Filter dort strich; leer: das Modell ließ es aus

    def als_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.ende, 3), "sprache_s": round(self.sprache_s, 3)}


def vereinigt(strecken: Iterable[Strecke]) -> list[Strecke]:
    aus: list[Strecke] = []
    for a, b in sorted(s for s in strecken if s[1] > s[0]):
        if aus and a <= aus[-1][1]:
            aus[-1] = (aus[-1][0], max(aus[-1][1], b))
        else:
            aus.append((a, b))
    return aus


def gehoert(segmente: Iterable[dict]) -> list[Strecke]:
    """Wo die Abschrift Wörter hat, ohne Polster; ein Segment ohne Wortzeiten gilt im Ganzen."""
    aus: list[Strecke] = []
    for s in segmente:
        worte = s.get("words") or []
        if worte:
            aus += [(w["start"], min(w["end"], w["start"] + WORT_HOECHSTENS_S)) for w in worte]
        else:
            aus.append((s["start"], s["end"]))
    return [(a, max(b, a + 0.01)) for a, b in aus]


def finden(sprache: Sequence[Strecke], segmente: Sequence[dict], aussortiert: Sequence[tuple[float, float, str]] = ()) -> list[Luecke]:
    """Lücken in der Reihenfolge der Aufnahme: je Zwischenraum zwischen zwei gehörten Wörtern, was darin gesprochen ist.
    `aussortiert`: Beginn, Ende und Grund dessen, was ein Filter strich."""
    if not sprache:
        return []
    worte = vereinigt(gehoert(segmente))
    raender = [(0.0, 0.0), *worte, (max(sprache[-1][1], worte[-1][1] if worte else 0.0),) * 2]
    aus = []
    for nr, ((_, a), (b, _)) in enumerate(zip(raender, raender[1:])):
        von = a + (WORT_POLSTER_S if nr else 0.0)                                  # Wortzeiten sind nicht genauer
        bis = b - (WORT_POLSTER_S if nr < len(raender) - 2 else 0.0)
        offen = sum(min(y, bis) - max(x, von) for x, y in sprache if y > von and x < bis)
        if offen < AUSSORTIERT_AB_S:
            continue
        grund = next((g for x, y, g in aussortiert if x < bis and y > von), "")
        if not grund and offen < LUECKE_AB_S:
            continue
        darin = [(max(x, a), min(y, b)) for x, y in sprache if y > a and x < b]
        aus.append(Luecke(max(a, darin[0][0] - WORT_POLSTER_S), min(b, darin[-1][1] + WORT_POLSTER_S) if nr < len(raender) - 2 else darin[-1][1] + WORT_POLSTER_S, offen, grund))
    return aus
