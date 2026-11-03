"""Musik mit einem Schein-Modell: Auftrag geht vollständig ans Modell, Teile werden zu einem Stück, die Datei ist
abholbar, Grenzen des Auftrags greifen. Das echte Modell läuft nur über die Leitstelle (siehe Handbuch)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from klangwerk import musik

from test_dienste import bereit, einstellungen

RATE = 44100


@dataclass
class Teil:
    audio: np.ndarray
    sample_rate: int = RATE


class ScheinModell:
    """Liefert zwei Teile zu je einer halben Sekunde Stereo, wie generate in mlx-audio Teile liefert."""

    def __init__(self):
        self.aufrufe = []

    def generate(self, **kw):
        self.aufrufe.append(kw)
        for _ in range(2):
            yield Teil(np.full((RATE // 2, 2), 0.1, dtype=np.float32))


class ScheinMotor(musik.Motor):
    def __init__(self):
        super().__init__("schein")
        self.modell_schein = ScheinModell()

    def laden(self):
        self._m = self.modell_schein
        return self._m


def test_musik_auftrag_ergebnis_und_abholen(tmp_path: Path):
    m = ScheinMotor()
    with TestClient(musik.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        r = c.post("/musik", json={"beschreibung": " Pop, warm ", "liedtext": "[Verse]\nÜber den Dächern\n", "dauer_s": 60,
                                   "schritte": 30, "startwert": 7})
        assert r.status_code == 200, r.text
        e = r.json()
        assert abs(e["duration"] - 1.0) < 0.01 and e["sample_rate"] == RATE and e["rechenzeit_s"] >= 0
        assert m.modell_schein.aufrufe == [{"text": "Pop, warm", "lyrics": "[Verse]\nÜber den Dächern", "steps": 30,
                                             "seed": 7, "duration": 60}]
        assert c.get(f"/tts/audio/{e['filename']}").status_code == 200


def test_musik_ohne_dauer_entscheidet_das_modell(tmp_path: Path):
    m = ScheinMotor()
    with TestClient(musik.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        assert c.post("/musik", json={"beschreibung": "Lo-Fi", "liedtext": "[instrumental]"}).status_code == 200
        assert "duration" not in m.modell_schein.aufrufe[0]


def test_musik_grenzen(tmp_path: Path):
    with TestClient(musik.erstellen(einstellungen(tmp_path), ScheinMotor())) as c:
        bereit(c)
        for falsch in ({"beschreibung": "", "liedtext": "x"}, {"beschreibung": "x", "liedtext": "x", "dauer_s": 301},
                       {"beschreibung": "x", "liedtext": "x", "schritte": 3},
                       {"beschreibung": "x", "liedtext": "x", "schritte": 31}):
            assert c.post("/musik", json=falsch).status_code == 422


def test_musik_steckbrief(tmp_path: Path):
    with TestClient(musik.erstellen(einstellungen(tmp_path), ScheinMotor())) as c:
        s = c.get("/musik/steckbrief").json()
        assert s["modell"] == "MiniMax-Music3" and s["gewichte"] == "mlx-community/MiniMax-Music3-mxfp8"
        assert s["hoechstdauer_s"] == 300 and s["schritte_hoechstens"] == 30 and s["referenzaudio"] is False
        assert s["lizenzhinweis"] == "MiniMax-Music3"
