"""Hören mit MOSS-Transcribe-Diarize mit einem Schein-Modell: Abschnitte mit Vorlauf, Zusammenführen nach Zeit, Sprecher
über Abschnitte (Vorlauf, Stimmabdruck), Formatfehler mit Wiederholung und Warnung, gleiche Schnittstelle wie der Dienst
hoeren. Das echte Modell läuft nur über die Leitstelle (siehe Handbuch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from klangwerk import hoeren_moss
from klangwerk.abschnitte import RATE, Abschnitt
from klangwerk.hoeren_moss import Beitrag, Bericht, Gehoert, Sprecherbuch, naht, zusammenfuehren

from test_dienste import bereit, einstellungen, wav_bytes


def gehoert(text: str) -> Gehoert:
    return Gehoert(text, 100, 50, 0.5, 1.2)


def stimme(n: int) -> np.ndarray:
    v = np.zeros(192, dtype=np.float32)
    v[n] = 1.0
    return v


class Schein:
    """Hört je Aufruf die nächste Antwort (relativ zum gehörten Stück); Stimmabdruck nach Zeit aus `stimmen`."""

    def __init__(self, antworten: list[str], stimmen: list[tuple[float, float, int]]):
        self.antworten, self.stimmen, self.stuecke, self.streng = list(antworten), stimmen, [], []

    def hoeren(self, stueck: np.ndarray, streng: bool) -> Gehoert:
        self.stuecke.append(len(stueck) / RATE)
        self.streng.append(streng)
        return gehoert(self.antworten.pop(0))

    def abdruecke(self, zeiten: list[dict]) -> list[list[float] | None]:
        aus = []
        for z in zeiten:
            treffer = [n for a, b, n in self.stimmen if a <= z["start"] < b]
            aus.append(stimme(treffer[0]).tolist() if treffer else None)
        return aus


TON = np.zeros(200 * RATE, dtype=np.float32)
SPRACHE = [(0.0, 5.0), (5.5, 59.0), (60.0, 63.0), (63.6, 110.0)]


def test_vorlauf_verbindet_sprecher_und_zaehlt_nicht_doppelt():
    # Abschnitt 1 bis 59,1 s: zwei Sprecher, der letzte Beitrag (S02, ab 50 s) ist der Vorlauf von Abschnitt 2.
    # Abschnitt 2 wird ab 49,9 s gehört; das Modell nennt dort den Vorlauf S01 und den neuen Sprecher S02.
    plan = [Abschnitt(0.0, 59.1), Abschnitt(59.9, 110.1)]
    s = Schein(["[0.0][S01]Hallo zusammen.[5.0][5.5][S01]Weiter geht es.[49.5][50.0][S02]Ich übernehme jetzt.[59.0]",
                "[0.1][S01]Ich übernehme jetzt.[9.1][10.1][S01]Und hier bin ich wieder.[13.1][13.7][S02]Neu dabei.[60.1]"],
               [(0.0, 49.9, 1), (50.0, 59.0, 2), (60.0, 63.5, 2), (63.6, 120, 3)])
    b = zusammenfuehren(TON, plan, SPRACHE, s.hoeren, s.abdruecke, "de", Sprecherbuch(0.5))
    assert [(round(x.start, 1), x.sprecher, x.text) for x in b.beitraege] == [
        (0.0, "S01", "Hallo zusammen."), (5.5, "S01", "Weiter geht es."), (50.0, "S02", "Ich übernehme jetzt."),
        (60.0, "S02", "Und hier bin ich wieder."), (63.6, "S03", "Neu dabei.")]
    assert s.stuecke[1] == round(110.1 - 49.9, 1) or abs(s.stuecke[1] - (110.1 - 49.9)) < 1e-3   # mit Vorlauf gehört
    zweiter = b.abschnitte[1]
    assert zweiter["vorlauf"] and zweiter["sprecher"]["S01"] == {"als": "S02", "wie": "vorlauf", "sicher": True}
    assert zweiter["sprecher"]["S02"]["wie"] == "neu" and b.warnungen == []


def test_sprecher_ueber_stimme_wiedererkannt_ohne_vorlauf():
    # Vorlauf zu lang (über 10 s): die Kennungen werden über den Stimmabdruck zugeordnet, auch vertauscht
    plan = [Abschnitt(0.0, 59.1), Abschnitt(59.9, 110.1)]
    s = Schein(["[0.0][S01]Erster Sprecher redet.[30.0][30.5][S02]Zweiter Sprecher antwortet lange.[59.0]",
                "[0.1][S01]Jetzt wieder der zweite.[20.0][20.5][S02]Und der erste.[50.0]"],
               [(0.0, 30.0, 1), (30.5, 59.0, 2), (60.0, 80.0, 2), (80.4, 120, 1)])
    b = zusammenfuehren(TON, plan, SPRACHE, s.hoeren, s.abdruecke, "de", Sprecherbuch(0.5))
    assert [x.sprecher for x in b.beitraege] == ["S01", "S02", "S02", "S01"]
    assert all(x.sicher for x in b.beitraege)
    assert s.stuecke[1] == plan[1].laenge or abs(s.stuecke[1] - plan[1].laenge) < 1e-3     # ohne Vorlauf


def test_zwei_kennungen_eines_abschnitts_nie_derselbe_sprecher():
    buch = Sprecherbuch(0.5)
    buch.zuordnen({"S01": stimme(1)}, {})
    z = buch.zuordnen({"S01": stimme(1), "S02": stimme(1) * 0.9 + stimme(2) * 0.1}, {})
    assert z["S01"].sprecher == "S01" and z["S02"].sprecher == "S02" and z["S02"].wie == "neu"


def test_naht_kuerzt_den_doppelt_gehoerten_anfang():
    # Befund 31.10.2026 (15-s-Abschnitte): das Modell beginnt mitten im Vorlauf und zieht ihn mit dem neuen Satz zusammen
    vorher = "Und wenn das wieder geht, dann kann ich einfach vom Sofa aus mit dem Handy die Steckdose anmachen und dann geht der Fernseher automatisch an."
    n = naht([Beitrag(236.0, 242.8, "S01", "Und wenn das wieder geht, dann kann ich die Steckdose"),
              Beitrag(239.8, 254.4, "S01", "anmachen und dann geht der Fernseher automatisch an. Das wird wieder schön."),
              Beitrag(254.6, 258.0, "S01", "Neu.")], 244.5, vorher)
    assert [x.text for x in n.doppelt] == ["Und wenn das wieder geht, dann kann ich die Steckdose"]
    assert [x.text for x in n.neu] == ["Das wird wieder schön.", "Neu."] and n.gekuerzt == 8
    assert n.neu[0].start == 244.5
    # ohne Wiederholung bleibt alles, ein einzelnes gleiches Wort gilt als Zufall
    n = naht([Beitrag(60.2, 70.0, "S02", "an der Stelle geht es weiter")], 60.0, "geht automatisch an")
    assert [x.text for x in n.neu] == ["an der Stelle geht es weiter"] and n.gekuerzt == 0


def test_formatfehler_einmal_streng_wiederholt_sonst_warnung():
    plan = [Abschnitt(0.0, 30.0)]
    s = Schein(["你好，这是一个测试", "[0.0][S01]Hallo, das ist ein Test.[3.0]"], [(0.0, 30.0, 1)])
    b = zusammenfuehren(TON, plan, [(0.0, 30.0)], s.hoeren, s.abdruecke, "de", Sprecherbuch(0.5))
    assert s.streng == [False, True] and [x.text for x in b.beitraege] == ["Hallo, das ist ein Test."]
    assert b.abschnitte[0]["wiederholt"] and b.abschnitte[0]["fehler"] is None and b.warnungen == []
    s = Schein(["nur Text ohne Marken", "wieder ohne Marken"], [(0.0, 30.0, 1)])
    b = zusammenfuehren(TON, plan, [(0.0, 30.0)], s.hoeren, s.abdruecke, "de", Sprecherbuch(0.5))
    assert [(x.sprecher, x.sicher, x.text) for x in b.beitraege] == [(None, False, "wieder ohne Marken")]
    assert b.warnungen and "keine Marken" in b.warnungen[0]


def test_notschnitt_wird_gemeldet():
    s = Schein(["[0.0][S01]Eins.[60.0]", "[0.0][S01]Zwei.[10.0]"], [(0.0, 200.0, 1)])
    b = zusammenfuehren(TON, [Abschnitt(0.0, 60.0, notschnitt=True), Abschnitt(60.0, 70.0)], [(0.0, 70.0)],
                        s.hoeren, s.abdruecke, "de", Sprecherbuch(0.5))
    assert any("Notschnitt" in w for w in b.warnungen)


def test_hoechstens_token_nach_dauer():
    assert hoeren_moss.hoechstens_token(60.0) == 60 * hoeren_moss.TOKEN_JE_S + hoeren_moss.TOKEN_ZUSCHLAG


def test_segmente_bereinigt_und_sortiert():
    segs = hoeren_moss.segmente([Beitrag(4.2, 7.9, "S02", "Und was fehlt noch?"), Beitrag(0.48, 4.0, "S01", " Hallo "),
                                 Beitrag(8.0, 8.5, "S01", "  "), Beitrag(4.2, 7.9, "S02", "Und was fehlt noch?"),
                                 Beitrag(9.0, 9.5, "S01", "Fachwörter: Qwen.")], "Fachwörter: Qwen.")
    assert [s["text"] for s in segs] == ["Hallo", "Und was fehlt noch?"]
    assert segs[0] == {"start": 0.48, "end": 4.0, "text": "Hallo", "words": [], "speaker": "S01", "speaker_sicher": True}


class ScheinMotor:
    def __init__(self):
        self.aufrufe = []

    def laden(self):
        return True

    def erkennen(self, pfad, hinweis="", sprache=None, abschnitt_s=60, schwelle=0.5):
        self.aufrufe.append({"pfad": pfad, "hinweis": hinweis, "sprache": sprache, "abschnitt_s": abschnitt_s,
                             "schwelle": schwelle})
        return Bericht([Beitrag(0.48, 4.0, "S01", "Hallo, wie ist der Stand?"), Beitrag(4.2, 7.9, "S02", "Gut.", False)],
                       [{"start": 0.0, "end": 8.0}], [])


def test_hoeren_moss_schnittstelle(tmp_path: Path):
    m = ScheinMotor()
    with TestClient(hoeren_moss.erstellen(einstellungen(tmp_path), m)) as c:
        bereit(c)
        assert c.get("/gesundheit").json()["dienst"] == "hoeren-moss"
        r = c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                   data={"language": "german", "diarize": "false", "prompt": "Fachwörter: Qwen."})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["model"] == "MOSS-Transcribe-Diarize 0.9B" and d["language"] == "de" and d["speakers"] == ["S01", "S02"]
        assert [s["speaker_sicher"] for s in d["segments"]] == [True, False] and d["abschnitte"] and d["warnungen"] == []
        assert m.aufrufe[-1]["abschnitt_s"] == hoeren_moss.ABSCHNITT_VORGABE_S and m.aufrufe[-1]["sprache"] == "de"
        assert not Path(m.aufrufe[-1]["pfad"]).exists()                                   # weggeräumt
        c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"abschnitt_s": "120"})
        assert m.aufrufe[-1]["abschnitt_s"] == 120
        assert c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"abschnitt_s": "600"}).status_code == 400
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                   data={"response_format": "verbose_json"})
        assert r.status_code == 200 and len(r.json()["segments"]) == 2
        assert c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"prompt": "x" * 1001}).status_code == 400


def test_abschnittslaenge_aus_der_dienstdatei(tmp_path: Path):
    m = ScheinMotor()
    e = einstellungen(tmp_path).model_copy(update={"abschnitt_s": 120.0, "abschnitt_hoechstens_s": 60.0})
    with TestClient(hoeren_moss.erstellen(e, m)) as c:
        bereit(c)
        c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert m.aufrufe[-1]["abschnitt_s"] == 60.0                     # Vorgabe nie über der sicheren Höchstlänge
        assert c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"abschnitt_s": "120"}).status_code == 400
        c.post("/stt", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"abschnitt_s": "30"})
        assert m.aufrufe[-1]["abschnitt_s"] == 30.0
