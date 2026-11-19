"""Regressionen an der Bibliotheksgrenze und messbare Dauer und Tonhöhe der fertigen Ausgabe."""

from __future__ import annotations

import ast
import io
from importlib.metadata import distribution
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from klangwerk import klonen, sprechen
from klangwerk.synthese import mit_tempo
from klangwerk.unterbau import Einstellungen
from test_dienste import bereit

RATE = 24000


@pytest.mark.parametrize("motorart", [sprechen.Motor, klonen.Motor])
def test_alle_teile_und_leerer_anfang(motorart):
    aufrufe = []

    def erzeugen(*, text, lang_code, ref_audio=None, ref_text=None):
        aufrufe.append((text, lang_code, ref_audio, ref_text))
        for audio in (None, np.array([]), np.array([0.1, 0.2]), np.array([0.3])):
            yield SimpleNamespace(audio=audio)

    motor = motorart("probe")
    motor.m = SimpleNamespace(generate=erzeugen)
    audio = motor.erzeugen(text="Erster Teil\nZweiter Teil", language="english", speed=1.2,
                          ref_audio="probe.wav", ref_text="Wortlaut der Probe")
    assert np.allclose(audio, [0.1, 0.2, 0.3])
    assert aufrufe == [("Erster Teil\nZweiter Teil", "english", "probe.wav", "Wortlaut der Probe")]
    motor.m = SimpleNamespace(generate=lambda **kw: iter([SimpleNamespace(audio=None), SimpleNamespace(audio=np.array([]))]))
    assert motor.erzeugen(text="Leer") is None


def test_sprachwahl_mit_installierter_erzeugungsfunktion():
    """Die installierte Funktion ausführen, nur ihre teuren internen Rechenfunktionen ersetzen."""
    pfad = distribution("mlx-audio").locate_file("mlx_audio/tts/models/qwen3_tts/qwen3_tts.py")
    baum = ast.parse(pfad.read_text())
    klasse = next(n for n in baum.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    funktion = next(n for n in klasse.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    modul = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), funktion],
                       type_ignores=[])
    raum = {"load_audio": lambda audio, **kw: audio}
    exec(compile(ast.fix_missing_locations(modul), str(pfad), "exec"), raum)
    for art, motorart in [("custom_voice", sprechen.Motor), ("base", klonen.Motor)]:
        for sprache in ("german", "english", "auto"):
            angekommen = []

            def erfassen(**kw):
                angekommen.append(kw)
                yield SimpleNamespace(audio=np.ones(3))

            modell = SimpleNamespace(config=SimpleNamespace(tts_model_type=art), sample_rate=RATE,
                                     speech_tokenizer=SimpleNamespace(has_encoder=True),
                                     generate_custom_voice=erfassen, _generate_icl=erfassen)
            modell.generate = lambda **kw: raum["generate"](modell, **kw)
            motor = motorart("probe")
            motor.m = modell
            assert len(motor.erzeugen(text="Prüfsatz", voice="aiden", language=sprache,
                                     ref_audio="probe", ref_text="Wortlaut der Probe")) == 3
            assert angekommen[0]["language"] == sprache


def ton(sekunden=3.0):
    t = np.arange(round(RATE * sekunden)) / RATE
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.mark.parametrize("tempo", [0.1, 0.5, 0.8, 1.0, 1.2, 2.0, 4.0])
async def test_tempo_dauer_und_tonhoehe(tempo):
    audio = await mit_tempo(ton(), RATE, tempo)
    assert len(audio) / RATE == pytest.approx(3.0 / tempo, rel=0.035, abs=0.06)
    ausschnitt = audio[len(audio) // 4:3 * len(audio) // 4]
    spektrum = np.abs(np.fft.rfft(ausschnitt * np.hanning(len(ausschnitt))))
    frequenz = np.fft.rfftfreq(len(ausschnitt), 1 / RATE)[np.argmax(spektrum)]
    assert frequenz == pytest.approx(440, abs=5)


@pytest.mark.parametrize("dienst", ["sprechen", "speech", "klonen"])
def test_tempo_am_endpunkt_und_unveraenderte_tokenbegrenzung(tmp_path, dienst):
    class Motor:
        sample_rate = RATE

        def __init__(self):
            self.aufrufe = []

        def laden(self):
            return True

        def erzeugen(self, **kw):
            self.aufrufe.append(kw)
            return ton()

    motor = Motor()
    app = (klonen if dienst == "klonen" else sprechen).erstellen(Einstellungen(ergebnisse=tmp_path), motor)
    with TestClient(app) as c:
        bereit(c)
        for tempo in (0.8, 1.0, 1.2):
            if dienst == "klonen":
                r = c.post("/tts/clone", data={"text": "Prüfsatz", "speed": str(tempo)},
                           files={"ref_audio": ("probe.wav", b"Probe", "audio/wav")})
            else:
                r = c.post("/tts" if dienst == "sprechen" else "/v1/audio/speech",
                           json={"text" if dienst == "sprechen" else "input": "Prüfsatz", "speed": tempo})
            assert r.status_code == 200, r.text
            wav = r.content if dienst == "speech" else c.get("/tts/audio/" + r.json()["filename"]).content
            audio, rate = sf.read(io.BytesIO(wav))
            assert rate == RATE and len(audio) / RATE == pytest.approx(3.0 / tempo, abs=0.06)
        assert len({kw["max_tokens"] for kw in motor.aufrufe}) == 1
