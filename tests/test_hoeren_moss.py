"""Hören mit MOSS-Transcribe-Diarize mit einem Schein-Modell: gleiche Schnittstelle wie der Dienst hoeren, dazu Sprecher je
Beitrag; Beiträge bereinigt, Hinweis als Hotwords. Das echte Modell läuft nur über die Leitstelle (siehe Handbuch)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from klangwerk import hoeren_moss
from klangwerk.hoeren_moss import Beitrag

from test_dienste import bereit, einstellungen, wav_bytes


class ScheinMoss:
    def __init__(self, beitraege: list[Beitrag]):
        self.beitraege, self.aufrufe = beitraege, []

    def laden(self):
        return True

    def erkennen(self, pfad: str, hinweis: str = ""):
        self.aufrufe.append({"pfad": pfad, "hinweis": hinweis})
        return self.beitraege


BEITRAEGE = [Beitrag(4.2, 7.9, "S02", "Und was fehlt noch?"), Beitrag(0.48, 4.0, "S01", " Hallo, wie ist der Stand? "),
             Beitrag(8.0, 8.5, "S01", "  "), Beitrag(4.2, 7.9, "S02", "Und was fehlt noch?")]


def test_segmente_bereinigt_und_sortiert():
    segs = hoeren_moss.segmente(BEITRAEGE)
    assert [s["text"] for s in segs] == ["Hallo, wie ist der Stand?", "Und was fehlt noch?"]
    assert segs[0] == {"start": 0.48, "end": 4.0, "text": "Hallo, wie ist der Stand?", "words": [], "speaker": "S01"}
    assert hoeren_moss.sprecherliste(segs) == ["S01", "S02"]


def test_segmente_ohne_wiederholten_hinweis_und_ende_nie_vor_beginn():
    segs = hoeren_moss.segmente([Beitrag(1.0, 0.5, "S01", "Fachwörter: Qwen, ComfyUI."), Beitrag(2.0, 1.0, None, "Gut so")],
                                "Fachwörter: Qwen, ComfyUI.")
    assert segs == [{"start": 2.0, "end": 2.0, "text": "Gut so", "words": [], "speaker": None}]


def test_hoechstens_token_nach_dauer():
    assert hoeren_moss.hoechstens_token(1.0) == hoeren_moss.TOKEN_MINDESTENS
    assert hoeren_moss.hoechstens_token(600.0) == 600 * hoeren_moss.TOKEN_JE_S


def test_anweisung_mit_hotwords():
    pytest.importorskip("moss_transcribe_diarize")
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    assert hoeren_moss.anweisung("") == DEFAULT_PROMPT
    assert hoeren_moss.anweisung("Qwen, Zigbee") == f"{DEFAULT_PROMPT}{hoeren_moss.HOTWORDS}Qwen, Zigbee"


def test_hoeren_moss_schnittstelle(tmp_path: Path):
    m = ScheinMoss(BEITRAEGE)
    with TestClient(hoeren_moss.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        assert c.get("/gesundheit").json()["dienst"] == "hoeren-moss"
        r = c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                   data={"language": "de", "diarize": "false", "prompt": "Fachwörter: Qwen."})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["model"] == "MOSS-Transcribe-Diarize 0.9B" and d["language"] == "de" and d["speakers"] == ["S01", "S02"]
        assert [s["speaker"] for s in d["segments"]] == ["S01", "S02"] and d["text"].startswith("Hallo, wie")
        assert m.aufrufe[-1]["hinweis"] == "Fachwörter: Qwen." and not Path(m.aufrufe[-1]["pfad"]).exists()   # weggeräumt
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                   data={"response_format": "verbose_json"})
        assert r.status_code == 200 and len(r.json()["segments"]) == 2
        assert c.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                      data={"response_format": "srt"}).status_code == 400
        assert c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"prompt": "x" * 1001}).status_code == 400
