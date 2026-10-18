"""Die fünf Dienste mit Schein-Modellen: Schnittstellen wie beim bisherigen Sprachdienst, übliches Format,
Nachbesserungen und Fehlerfälle. Echte Modelle laufen nur in test_echt.py (gezielt mit -m echt)."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from klangwerk import hoeren, klonen, laenge, sprechen, sprecher, trennen
from klangwerk.unterbau import Einstellungen, Rechner, ZugangFehlt

RATE = 24000


def einstellungen(tmp: Path, **mehr) -> Einstellungen:
    return Einstellungen(ergebnisse=tmp / "ergebnisse", **mehr)


def bereit(c: TestClient) -> None:
    for _ in range(100):
        if c.get("/gesundheit").status_code == 200:
            return
        time.sleep(0.02)
    raise AssertionError(c.get("/gesundheit").json())


def wav_bytes(sekunden: float = 1.0, rate: int = 16000) -> bytes:
    puffer = io.BytesIO()
    t = np.arange(int(sekunden * rate)) / rate
    sf.write(puffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate, format="WAV")
    return puffer.getvalue()


# ── sprechen ─────────────────────────────────────────────────────────────
class ScheinSprechen:
    sample_rate = RATE

    def __init__(self):
        self.aufrufe = []

    def laden(self):
        return True

    def stimmen(self):
        return ["aiden", "serena"]

    def sprachen(self):
        return ["german", "english"]

    def erzeugen(self, **kw):
        self.aufrufe.append(kw)
        return None if len(self.aufrufe) == 1 else np.zeros(RATE, dtype=np.float32)   # erster Versuch leer


def test_sprechen(tmp_path: Path):
    m = ScheinSprechen()
    with TestClient(sprechen.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        r = c.post("/tts", json={"text": "Hallo Welt", "voice": "aiden", "instruct": " ruhig "})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["duration"] == 1.0 and d["sample_rate"] == RATE and d["voice"] == "aiden"
        assert len(m.aufrufe) == 2 and m.aufrufe[1]["instruct"] == "ruhig"           # Wiederholung nach leerem Versuch
        assert m.aufrufe[1]["max_tokens"] == laenge.hoechstens_token(m.aufrufe[1]["text"], m.aufrufe[1].get("speed", 1.0))
        assert c.get(f"/tts/audio/{d['filename']}").content[:4] == b"RIFF"
        assert c.get("/tts/audio/..%2Fgeheim.wav").status_code == 404
        r = c.post("/v1/audio/speech", json={"input": "Hallo", "voice": "serena"})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
        assert c.post("/v1/audio/speech", json={"input": "x", "response_format": "mp3"}).status_code == 400
        assert c.post("/tts", json={"text": "x", "model": "1.7b-voicedesign-4bit"}).status_code == 409
        assert c.post("/tts", json={"text": "x", "model": "1.7b-customvoice-8bit"}).status_code == 200
        assert c.get("/tts/voices").json() == {"speakers": ["aiden", "serena"], "languages": ["german", "english"]}
        assert c.get("/health").json()["dienst"] == "sprechen"


def test_sprechen_gibt_nach_drei_leeren_versuchen_auf(tmp_path: Path):
    class Leer(ScheinSprechen):
        def erzeugen(self, **kw):
            self.aufrufe.append(kw)
            return None
    m = Leer()
    with TestClient(sprechen.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        assert c.post("/tts", json={"text": "Hallo"}).status_code == 500 and len(m.aufrufe) == 3


# ── klonen ───────────────────────────────────────────────────────────────
def test_artefakt_entfernen():
    sr = 16000
    laut = 0.5 * np.ones(int(0.2 * sr), dtype=np.float32)
    pause = np.zeros(int(0.3 * sr), dtype=np.float32)
    rede = 0.5 * np.ones(sr, dtype=np.float32)
    ohne, info = klonen.artefakt_entfernen(np.concatenate([laut, pause, rede]), sr)
    assert info is not None and abs(len(ohne) / sr - 1.0) < 0.02
    unveraendert, info = klonen.artefakt_entfernen(np.concatenate([pause, rede]), sr)   # beginnt mit Stille
    assert info is None and len(unveraendert) == len(pause) + len(rede)
    lang = 0.5 * np.ones(int(0.8 * sr), dtype=np.float32)                                  # langer Anfang ist Rede
    assert klonen.artefakt_entfernen(np.concatenate([lang, pause, rede]), sr)[1] is None


def test_klonen(tmp_path: Path):
    class Schein:
        sample_rate = 16000
        aufrufe = []

        def laden(self):
            return True

        def erzeugen(self, **kw):
            self.aufrufe.append(kw)
            assert Path(kw["ref_audio"]).suffix == ".mp3"                                  # Endung bleibt erhalten
            return np.concatenate([0.5 * np.ones(3200), np.zeros(4800), 0.5 * np.ones(16000)]).astype(np.float32)
    m = Schein()
    with TestClient(klonen.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        r = c.post("/tts/clone", data={"text": "Hallo", "ref_text": "Referenz"}, files={"ref_audio": ("stimme.mp3", b"ID3...", "audio/mpeg")})
        assert r.status_code == 200, r.text
        assert abs(r.json()["duration"] - 1.0) < 0.02                                       # Artefakt entfernt
        assert m.aufrufe[0]["ref_text"] == "Referenz" and not Path(m.aufrufe[0]["ref_audio"]).exists()   # aufgeräumt
        assert m.aufrufe[0]["max_tokens"] == laenge.hoechstens_token("Hallo")                     # nie bis 4096 durchlaufen
        # ohne Text: reiner Abdruck (ref_text None), nie ICL mit leerem Text; ausdrücklich abdruck trotz Text
        assert c.post("/tts/clone", data={"text": "Hallo"}, files={"ref_audio": ("s.mp3", b"ID3", "audio/mpeg")}).status_code == 200
        assert m.aufrufe[-1]["ref_text"] is None
        c.post("/tts/clone", data={"text": "Hallo", "ref_text": "Referenz", "modus": "abdruck"},
               files={"ref_audio": ("s.mp3", b"ID3", "audio/mpeg")})
        assert m.aufrufe[-1]["ref_text"] is None
        assert c.post("/tts/clone", data={"text": "Hallo", "modus": "icl"}, files={"ref_audio": ("s.mp3", b"ID3", "audio/mpeg")}).status_code == 400
        assert c.post("/tts/clone", data={"text": "x"}, files={"ref_audio": ("a.wav", b"", "audio/wav")}).status_code == 400
        assert c.post("/tts/clone", data={"text": " "}, files={"ref_audio": ("a.wav", b"x", "audio/wav")}).status_code == 400
        assert c.post("/tts/clone", data={"text": "x", "speed": "9"}, files={"ref_audio": ("a.wav", b"x", "audio/wav")}).status_code == 400


# ── hoeren ───────────────────────────────────────────────────────────────
ROH = {"language": "de", "segments": [
    {"start": 0.0, "end": 2.0, "text": " Guten Morgen ", "words": [{"word": " Guten", "start": 0.0, "end": 0.5}, {"word": " Morgen", "start": 0.5, "end": 1.0}]},
    {"start": 2.0, "end": 3.0, "text": " Guten Morgen ", "words": []},                       # Wiederholung
    {"start": 3.0, "end": 4.0, "text": "Untertitel von XY", "words": []},                     # Halluzination
    {"start": 4.0, "end": 5.0, "text": "ja ja ja ja ja ja ja ja", "words": []},              # fast nur ein Wort
    {"start": 5.1, "end": 7.0, "text": "wie geht es", "words": [{"word": " wie", "start": 5.1, "end": 5.4}]},
    {"start": 7.5, "end": 9.0, "text": "dir heute", "words": []},
]}


class ScheinHoeren:
    def __init__(self):
        self.aufrufe = []

    def laden(self):
        return True

    def erkennen(self, pfad, sprache):
        self.aufrufe.append(sprache)
        return ROH

    def abschnitte(self, pfad):
        return [{"start": 0.0, "end": 0.5}, {"start": 0.6, "end": 0.7}, {"start": 1.0, "end": 1.5}]   # mittlerer zu kurz


def test_segmente_bereinigen():
    s = hoeren.segmente_bereinigen(ROH["segments"])
    assert [x["text"] for x in s] == ["Guten Morgen wie geht es", "dir heute"]
    assert s[0]["start"] == 0.0 and s[0]["end"] == 7.0 and [w["word"] for w in s[0]["words"]] == [" Guten", " Morgen", " wie"]


def test_hoeren(tmp_path: Path):
    m = ScheinHoeren()
    with TestClient(hoeren.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        r = c.post("/stt", data={"language": "german", "diarize": "true"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        d = r.json()
        assert r.status_code == 200 and d["text"] == "Guten Morgen wie geht es dir heute" and d["speakers"] == []
        assert d["model"] == "Whisper Large V3 (MLX)" and m.aufrufe[-1] == "de"
        r = c.post("/v1/audio/transcriptions", data={"response_format": "json"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert r.json() == {"text": "Guten Morgen wie geht es dir heute"} and m.aufrufe[-1] is None   # ohne Sprache: erkennen
        r = c.post("/v1/audio/transcriptions", data={"response_format": "text", "language": "en"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert r.text == "Guten Morgen wie geht es dir heute" and m.aufrufe[-1] == "en"
        r = c.post("/v1/audio/transcriptions", data={"response_format": "verbose_json"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert len(r.json()["segments"]) == 2
        assert c.post("/v1/audio/transcriptions", data={"response_format": "srt"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")}).status_code == 400
        assert c.post("/stt", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400


def test_hoeren_strom(tmp_path: Path):
    m = ScheinHoeren()
    with TestClient(hoeren.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        r = c.post("/stt/stream", data={"language": "german"}, files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")})
        ereignisse = [json.loads(z[6:]) for z in r.text.splitlines() if z.startswith("data: ")]
        arten = [e["type"] for e in ereignisse]
        assert arten[0] == "start" and arten[-1] == "done" and arten.count("segment") == 4     # zwei Abschnitte, je zwei Segmente
        zweiter = [e for e in ereignisse if e["type"] == "segment"][2]["segment"]
        assert zweiter["start"] == 1.0 and zweiter["words"][0]["start"] == 1.0               # absolute Zeiten
        assert ereignisse[-1]["stats"]["total_chunks"] == 2 and len(m.aufrufe) == 2


# ── sprecher ─────────────────────────────────────────────────────────────
def test_sprecher(tmp_path: Path):
    class Schein:
        def laden(self):
            return True

        def abdruecke(self, pfad, abschnitte):
            return [None if s["end"] - s["start"] < 1 else [0.1] * 192 for s in abschnitte]
    with TestClient(sprecher.erstellen(einstellungen(tmp_path), Schein())) as c:
        bereit(c)
        r = c.post("/speaker/embed", data={"segments": json.dumps([{"start": 0, "end": 2}, {"start": 2, "end": 2.5}])},
                   files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert r.status_code == 200 and r.json()["dim"] == 192 and r.json()["embeddings"][1] is None
        assert c.post("/speaker/embed", data={"segments": "{}"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")}).status_code == 400


# ── trennen ──────────────────────────────────────────────────────────────
def test_trennen_ohne_zugang_dann_mit(tmp_path: Path):
    class Schein:
        def __init__(self):
            self.zugaenge = []

        def laden(self, zugang=None):
            self.zugaenge.append(zugang)
            if not zugang:
                raise ZugangFehlt("Zugang fehlt")
            return True

        def trennen(self, pfad, anzahl):
            return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    m = Schein()
    with TestClient(trennen.erstellen(einstellungen(tmp_path), m)) as c:
        for _ in range(100):
            if c.get("/gesundheit").json()["zustand"] != "laedt":
                break
            time.sleep(0.02)
        g = c.get("/gesundheit")
        assert g.status_code == 503 and g.json()["zustand"] == "ohne_zugang"
        assert c.post("/speaker/diarize", files={"file": ("a.wav", wav_bytes(), "audio/wav")}).status_code == 503
        r = c.post("/speaker/diarize", data={"hf_token": "hf_probe"}, files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert r.status_code == 200 and r.json()["turns"][0]["speaker"] == "SPEAKER_00" and m.zugaenge[-1] == "hf_probe"
        assert c.get("/gesundheit").status_code == 200


# ── unterbau ─────────────────────────────────────────────────────────────
async def test_eine_rechnung_zugleich_je_prozess(tmp_path: Path):
    r = Rechner("probe", "modell", einstellungen(tmp_path))
    await r.laden(lambda: True)
    gleichzeitig, hoechst, sperre = [0], [0], threading.Lock()

    def arbeit():
        with sperre:
            gleichzeitig[0] += 1
            hoechst[0] = max(hoechst[0], gleichzeitig[0])
        time.sleep(0.05)
        with sperre:
            gleichzeitig[0] -= 1
        return True

    await asyncio.gather(*(r.rechnen(arbeit) for _ in range(5)))
    assert hoechst[0] == 1


async def test_rechnen_vor_dem_laden_wird_abgelehnt(tmp_path: Path):
    from fastapi import HTTPException
    r = Rechner("probe", "modell", einstellungen(tmp_path))
    with pytest.raises(HTTPException) as f:
        await r.rechnen(lambda: True)
    assert f.value.status_code == 503


def test_ton_rechnet_mit_ffmpeg_ohne_modell(tmp_path):
    """Der Tondienst (Appkern-Routen im Klangwerk): bereit ohne Modell, Messwerte und Ausgaben, einmal abholbar."""
    from klangwerk import ton

    welle = (0.5 * np.sin(2 * np.pi * 440 * np.arange(RATE * 2) / RATE)).astype(np.float32)
    quelle = tmp_path / "quelle.wav"
    sf.write(str(quelle), np.concatenate([np.zeros(RATE // 2, np.float32), welle]), RATE)
    with TestClient(ton.erstellen(einstellungen(tmp_path))) as c:
        bereit(c)
        assert c.get("/gesundheit").json()["modell"] == "ffmpeg"
        auftrag = {"ausgaben": [{"format": {"rate": 16000, "kanaele": 1, "kodierung": "pcm_s16le"}}],
                   "messen": {"laut": {}}}
        with quelle.open("rb") as f:
            r = c.post("/ton/aufbereiten", files={"datei": ("quelle.wav", f, "audio/wav")}, data={"auftrag": json.dumps(auftrag)})
        assert r.status_code == 200, r.text
        erg = r.json()
        assert abs(erg["messwerte"]["laut"]["von_s"] - 0.46) < 0.01 and abs(erg["ausgaben"][0]["dauer_s"] - 2.5) < 0.01
        name = erg["ausgaben"][0]["name"]
        datei = c.get(f"/ton/ergebnis/{name}")
        assert datei.status_code == 200 and sf.info(io.BytesIO(datei.content)).samplerate == 16000
        assert c.get(f"/ton/ergebnis/{name}").status_code == 404
        kaputt = c.post("/ton/aufbereiten", files={"datei": ("k.wav", b"kein Ton", "audio/wav")}, data={"auftrag": "{}"})
        assert kaputt.status_code == 422 and kaputt.json()["error"]["code"] == "ton_fehler"


def test_hoechstlaenge_nach_text():
    """Großzügig (5 Zeichen je Sekunde und 3 s Rand), langsameres Tempo verlängert, nie über die Vorgabe des Modells."""
    assert laenge.hoechstens_token("Hallo") == 48                                    # (3 + 1) s * 12
    assert laenge.hoechstens_token("x" * 180) == (3 + 36) * 12                       # ein langes Stück: 39 s
    assert laenge.hoechstens_token("x" * 180, tempo=0.5) == (3 + 36) * 24
    assert laenge.hoechstens_token("x" * 100000) == laenge.HOECHSTENS_TOKEN
