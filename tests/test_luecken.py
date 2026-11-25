"""Gesprochenes fällt nie still weg (BT-91): Lücken finden, nachhören, benennen. Alles mit einem nachgestellten Motor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from klangwerk import hoeren, luecken
from tests.test_dienste import bereit, einstellungen, wav_bytes


def segment(von: float, bis: float, text: str, **mehr) -> dict:
    """Ein Segment mit gleichmäßig verteilten Wörtern."""
    worte = text.split()
    je = (bis - von) / len(worte)
    return {"start": von, "end": bis, "text": f" {text}", "words": [{"word": f" {w}", "start": von + i * je, "end": von + (i + 1) * je} for i, w in enumerate(worte)], **mehr}


def satz(n: int, vorsilbe: str = "wort") -> str:
    return " ".join(f"{vorsilbe}{i}" for i in range(n))


# ── luecken ──────────────────────────────────────────────────────────────
def test_vereinigen():
    assert luecken.vereinigt([(3, 4), (1, 2), (1.5, 3)]) == [(1, 4)]


def test_alles_gehoert_keine_luecke():
    sprache = [(0.0, 10.0), (10.6, 20.0)]
    assert luecken.finden(sprache, [segment(0, 10, satz(25)), segment(10.6, 20, satz(25))]) == []
    assert luecken.finden([], [segment(0, 1, "ja")]) == [] and luecken.finden([], []) == []


def test_ausgelassene_strecke_wird_gefunden_und_beruehrt_keine_woerter():
    sprache = [(0.0, 44.7), (46.1, 54.1), (54.6, 63.2), (64.4, 70.0)]
    gehoert = [segment(0, 44.7, satz(120)), segment(65.1, 70, satz(12))]
    [l] = luecken.finden(sprache, gehoert)
    assert 44.7 <= l.start <= 46.1 and l.ende == 65.1 and 16.5 < l.sprache_s < 17.5 and l.grund == ""      # bis an das nächste Wort


def test_ein_gezogenes_segment_deckt_nur_seine_woerter():
    """Das Modell nennt ein Segment über 20 s, seine drei Wörter liegen am Ende."""
    s = {"start": 45.0, "end": 66.0, "text": " drei Wörter nur", "words": [{"word": " drei", "start": 64.5, "end": 65}, {"word": " Wörter", "start": 65, "end": 65.5}, {"word": " nur", "start": 65.5, "end": 66}]}
    [l] = luecken.finden([(45.0, 66.0)], [s])
    assert l.start == pytest.approx(45.0, abs=0.36) and l.ende <= 64.5
    lang = {"start": 45.0, "end": 66.0, "text": " nur", "words": [{"word": " nur", "start": 45.0, "end": 66.0}]}      # ein "Wort" über 21 s
    assert luecken.finden([(45.0, 66.0)], [lang])[0].sprache_s > 18


def test_kurze_laute_sind_keine_luecke_ausser_ein_filter_strich_dort():
    sprache, gehoert = [(0.0, 5.0), (6.0, 6.6), (8.0, 12.0)], [segment(0, 5, satz(12)), segment(8, 12, satz(10))]
    assert luecken.finden(sprache, gehoert) == []
    [l] = luecken.finden(sprache, gehoert, [(6.0, 6.6, "wiederholt")])
    assert l.grund == "wiederholt" and l.start <= 6.0 and l.ende >= 6.6


def test_am_anfang_und_am_ende():
    a, e = luecken.finden([(0.5, 4.0), (5.0, 10.0), (11.0, 14.0)], [segment(5, 10, satz(12))])
    assert (a.start, a.ende) == (pytest.approx(0.15), pytest.approx(4.35)) and (e.start, e.ende) == (pytest.approx(10.65), pytest.approx(14.35))
    [l] = luecken.finden([(0.5, 4.0)], [])
    assert (l.start, l.ende, l.sprache_s) == (pytest.approx(0.15), pytest.approx(4.35), pytest.approx(3.5))


def test_ohne_wortzeiten_gilt_das_segment_im_ganzen():
    assert luecken.finden([(0.0, 10.0)], [{"start": 0, "end": 10, "text": "ohne Zeiten", "words": []}]) == []


# ── nachhören ────────────────────────────────────────────────────────────
class Vergesslich:
    """Ein Motor, der beim ersten Hören etwas verliert und beim Nachhören antwortet, was er soll."""

    def __init__(self, sprache, erstes, nach):
        self._sprache, self._erstes, self._nach, self.nachgefragt = sprache, erstes, nach, []

    def laden(self):
        return True

    def ton(self, pfad):
        return np.zeros(10, dtype=np.float32)

    def stellen(self, ton):
        return self._sprache

    def erkennen(self, ton, sprache, hinweis="", abschnitt_s=None, stellen=()):
        return {"segments": self._erstes, "language": "de"}

    def nachhoeren(self, ton, von, bis, sprache, hinweis, stellen, fenster_s):
        self.nachgefragt.append((von, bis, sprache, hinweis, fenster_s))
        return self._nach(von, bis, hinweis, fenster_s)

    def abschnitte(self, pfad, hoechstens_s=15):
        return [{"start": 0.0, "end": 20.0}]


SPRACHE = [(0.0, 10.0), (10.6, 30.0), (31.0, 40.0)]


def test_verlorene_strecke_kommt_zurueck():
    m = Vergesslich(SPRACHE, [segment(0, 10, satz(25, "a")), segment(31, 40, satz(22, "c"))], lambda von, bis, h, f: [segment(10.6, 20, satz(22, "b")), segment(20, 30, satz(22, "d"))])
    e = hoeren.vollstaendig(m, "x", None, "Fachwörter: Qwen.")
    text = " ".join(s["text"] for s in e["segments"])
    assert "b0" in text and "d21" in text and text.index("a24") < text.index("b0") < text.index("d21") < text.index("c0")
    assert e["luecken"] == [] and [(n["grund"], n["woerter"]) for n in e["nachgehoert"]] == [("ausgelassen", 44)]
    [(von, bis, sprache, hinweis, fenster)] = m.nachgefragt                   # einmal genügt, mit Hinweis und erkannter Sprache
    assert 10.0 <= von <= 10.6 and 30.0 <= bis <= 31.0 and (sprache, hinweis, fenster) == ("de", "Fachwörter: Qwen.", 12.0)


@pytest.mark.parametrize("gestrichen,grund", [
    ("Qwen, ComfyUI.", "hinweis"), ("ja ja ja ja ja ja ja ja", "schleife"), ("Vielen Dank für das Zuschauen", "phrase"), (satz(25, "a"), "wiederholt")])
def test_was_ein_filter_strich_wird_nachgehoert(gestrichen, grund):
    """Über echter Sprache streicht kein Filter ersatzlos."""
    m = Vergesslich(SPRACHE, [segment(0, 10, satz(25, "a")), segment(10.6, 30, gestrichen), segment(31, 40, satz(22, "c"))],
                    lambda von, bis, h, f: [segment(10.6, 30, satz(45, "b"))])
    e = hoeren.vollstaendig(m, "x", "de", "Fachwörter: Qwen, ComfyUI.")
    assert "b44" in " ".join(s["text"] for s in e["segments"]) and e["luecken"] == [] and e["nachgehoert"][0]["grund"] == grund


def test_zweiter_versuch_ohne_hinweis_in_kleineren_fenstern():
    """Wiederholt Whisper beim Nachhören wieder nur den Hinweis, hört es ohne ihn."""
    m = Vergesslich(SPRACHE, [segment(0, 10, satz(25, "a")), segment(31, 40, satz(22, "c"))],
                    lambda von, bis, h, f: [segment(10.6, 30, "Qwen, ComfyUI.")] if h else [segment(10.6, 30, satz(45, "b"))])
    e = hoeren.vollstaendig(m, "x", "de", "Fachwörter: Qwen, ComfyUI.")
    assert [(x[3], x[4]) for x in m.nachgefragt] == [("Fachwörter: Qwen, ComfyUI.", 12.0), ("", 6.0)] and e["luecken"] == []


def test_zuletzt_mit_umfeld_und_nur_die_woerter_in_der_luecke():
    """Für sich allein hört das Modell ein kurzes Stück nicht, mit dem Satz ringsum schon."""
    sprache = [(0.0, 10.0), (10.6, 13.0), (14.0, 30.0)]
    erstes = [segment(0, 10, satz(25, "a")), segment(14, 30, satz(40, "c"))]
    ringsum = segment(8.0, 16.0, satz(16, "r"))                                   # je Wort 0,5 s
    m = Vergesslich(sprache, erstes, lambda von, bis, h, f: [ringsum] if von < 7 else [])
    e = hoeren.vollstaendig(m, "x", "de")
    text = " ".join(s["text"] for s in e["segments"])
    drin = [i for i in range(16) if f"r{i}" in text.split()]
    assert drin and all(10.0 <= 8.0 + i * 0.5 and 8.5 + i * 0.5 <= 14.0 for i in drin) and e["luecken"] == []
    assert m.nachgefragt[-1][0] == pytest.approx(m.nachgefragt[0][0] - 4.0) and m.nachgefragt[-1][4] == 20.0


def test_unsicheres_wird_nie_text_sondern_benannt():
    for unsicher in ({"avg_logprob": -1.6}, {"compression_ratio": 3.4}, {"no_speech_prob": 0.9, "avg_logprob": -0.8}):
        m = Vergesslich(SPRACHE, [segment(0, 10, satz(25, "a")), segment(31, 40, satz(22, "c"))], lambda von, bis, h, f, u=unsicher: [segment(10.6, 30, satz(45, "b"), **u)])
        e = hoeren.vollstaendig(m, "x", "de")
        assert "b0" not in str(e["segments"]) and e["nachgehoert"] == [] and len(m.nachgefragt) == 3
        [l] = e["luecken"]
        assert 10.0 <= l["start"] <= 10.6 and 30.0 <= l["end"] <= 31.0 and l["sprache_s"] > 18


def test_dasselbe_zweimal_gesprochen_bleibt_zweimal():
    """Ein kurzes Ja an zwei Stellen: früher fiel das zweite weg."""
    sprache = [(0.0, 5.0), (6.0, 6.6), (7.0, 12.0), (13.0, 13.6), (14.0, 20.0)]
    erstes = [segment(0, 5, satz(12, "a")), segment(6, 6.6, "Ja, genau."), segment(7, 12, satz(12, "b")), segment(13, 13.6, "Ja, genau."), segment(14, 20, satz(14, "c"))]
    m = Vergesslich(sprache, erstes, lambda von, bis, h, f: [segment(13, 13.6, "Ja, genau.")])
    assert " ".join(s["text"] for s in hoeren.vollstaendig(m, "x", "de")["segments"]).count("Ja, genau.") == 2


def test_ein_dank_wird_behalten_wenn_er_zweimal_gehoert_ist_und_sonst_nicht():
    sprache = [(0.0, 5.0), (6.0, 8.0)]
    dank = "Vielen Dank für deine Nachricht."
    beide = Vergesslich(sprache, [segment(0, 5, satz(12)), segment(6, 8, dank)], lambda *a: [segment(6, 8, dank)])
    assert dank in hoeren.vollstaendig(beide, "x", "de")["segments"][0]["text"]
    stille = Vergesslich([(0.0, 5.0)], [segment(0, 5, satz(12)), segment(6, 8, dank)], lambda *a: pytest.fail("in Stille wird nicht nachgehört"))
    e = hoeren.vollstaendig(stille, "x", "de")
    assert dank not in str(e["segments"]) and e["luecken"] == []


def test_dienst_nennt_nachgehoertes_und_luecken(tmp_path: Path):
    m = Vergesslich(SPRACHE, [segment(0, 10, satz(25, "a"))], lambda von, bis, h, f: [segment(10.6, 30, satz(45, "b"))] if von < 12 else [])
    with TestClient(hoeren.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        d = c.post("/stt", data={"language": "german"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")}).json()
        assert d["nachgehoert"] and "b44" in d["text"] and len(d["luecken"]) <= 1
        d = c.post("/v1/audio/transcriptions", data={"response_format": "verbose_json"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")}).json()
        assert "luecken" in d
        r = c.post("/stt/stream", data={"language": "german"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert '"luecken"' in r.text and r.text.count('"type": "segment"') >= 1


def test_motor_hoert_nur_die_strecke_mit_ihren_sprachstellen(monkeypatch):
    """Der echte Motor ohne Modell: Ausschnitt, Fenster und Zeiten."""
    m = hoeren.Motor("x")
    gehoert = []

    def whisper(ton, sprache, hinweis):
        gehoert.append((len(ton) / hoeren.RATE, sprache, hinweis))
        return {"language": "de", "segments": [segment(0.1, 1.0, "zwei Wörter")]}
    monkeypatch.setattr(m, "_hoeren", whisper)
    ton = np.zeros(60 * hoeren.RATE, dtype=np.float32)
    s = m.nachhoeren(ton, 20.0, 40.0, "de", "Hinweis", [(0.0, 19.0), (20.5, 29.0), (30.0, 39.5), (45.0, 50.0)], 12.0)
    assert len(gehoert) == 2 and all(5 < d <= 12 and (sp, h) == ("de", "Hinweis") for d, sp, h in gehoert)
    assert [round(x["start"], 1) for x in s] == [20.5, 30.0] and s[1]["words"][0]["start"] == pytest.approx(30.0, abs=0.05)
    assert m.erkennen(ton, "de")["abschnitte"] == [] and gehoert[-1][0] == 60.0                # ohne Hinweis: die ganze Datei
    assert len(m.erkennen(ton, "de", "Hinweis", None, [(0.0, 20.0), (25.0, 50.0)])["abschnitte"]) == 2
