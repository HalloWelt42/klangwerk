"""Mit echten Modellen (nur auf dem Rechner, auf dem sie liegen): pytest -m echt. Ohne Netz (Modelle aus dem Zwischenspeicher).

Kette: sprechen erzeugt einen Satz, hoeren erkennt ihn wieder, klonen spricht mit dieser Stimme einen neuen Satz,
sprecher bildet Stimmabdrücke beider Aufnahmen. trennen nur mit KLANGWERK_HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from klangwerk import hoeren, klonen, sprechen, sprecher, trennen
from klangwerk.unterbau import Einstellungen

pytestmark = pytest.mark.echt
os.environ.setdefault("HF_HUB_OFFLINE", "1")
SATZ = "Guten Morgen, das Klangwerk spricht mit eigener Stimme."


def bereit(c: TestClient, sekunden: float = 300) -> None:
    ende = time.time() + sekunden
    while time.time() < ende:
        g = c.get("/gesundheit")
        if g.status_code == 200:
            return
        assert g.json()["zustand"] == "laedt", g.json()
        time.sleep(0.5)
    raise AssertionError("Modell nicht rechtzeitig geladen")


@pytest.fixture(scope="module")
def ablage(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("echt")


def test_kette(ablage: Path):
    e = Einstellungen(ergebnisse=ablage / "ergebnisse")
    with TestClient(sprechen.erstellen(e)) as c:
        bereit(c)
        r = c.post("/v1/audio/speech", json={"input": SATZ, "voice": "aiden", "language": "german"})
        assert r.status_code == 200 and r.content[:4] == b"RIFF"
        (ablage / "satz.wav").write_bytes(r.content)
    with TestClient(hoeren.erstellen(e)) as c:
        bereit(c)
        with open(ablage / "satz.wav", "rb") as f:
            text = c.post("/v1/audio/transcriptions", data={"language": "de"}, files={"file": ("satz.wav", f, "audio/wav")}).json()["text"]
        assert "Klangwerk" in text or "Klang" in text, text
    with TestClient(klonen.erstellen(e)) as c:
        bereit(c)
        with open(ablage / "satz.wav", "rb") as f:
            r = c.post("/tts/clone", data={"text": "Und jetzt mit geklonter Stimme.", "ref_text": SATZ}, files={"ref_audio": ("satz.wav", f, "audio/wav")})
        assert r.status_code == 200, r.text
        (ablage / "klon.wav").write_bytes(c.get(f"/tts/audio/{r.json()['filename']}").content)
    with TestClient(sprecher.erstellen(e)) as c:
        bereit(c)
        abdruecke = []
        for name in ("satz.wav", "klon.wav"):
            with open(ablage / name, "rb") as f:
                r = c.post("/speaker/embed", data={"segments": json.dumps([{"start": 0, "end": 2.5}])}, files={"file": (name, f, "audio/wav")})
            abdruecke.append(np.array(r.json()["embeddings"][0]))
        aehnlichkeit = float(abdruecke[0] @ abdruecke[1])
        assert aehnlichkeit > 0.3, aehnlichkeit                                  # geklonte Stimme ähnelt der Vorlage


@pytest.mark.skipif(not os.environ.get("KLANGWERK_HF_TOKEN"), reason="Sprechertrennung braucht KLANGWERK_HF_TOKEN")
def test_trennen(ablage: Path):
    with TestClient(trennen.erstellen(Einstellungen(ergebnisse=ablage / "ergebnisse"))) as c:
        bereit(c)
        with open(ablage / "satz.wav", "rb") as f:
            assert c.post("/speaker/diarize", files={"file": ("satz.wav", f, "audio/wav")}).json()["turns"]
