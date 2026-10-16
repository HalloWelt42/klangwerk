"""Dienst trennen: wer spricht wann, mit pyannote/speaker-diarization-3.1 (Grafik über torch, wenn vorhanden).

Das Modell ist gesperrt: es braucht einen Zugang (KLANGWERK_HF_TOKEN in der Dienstdatei oder die Anmeldung von
huggingface_hub auf dem Rechner). Ohne Zugang läuft der Dienst im Zustand `ohne_zugang`; ein Aufruf mit `hf_token`
(wie beim bisherigen Sprachdienst) lädt das Modell nach. Antwort 503, solange es nicht ladbar ist; Aufrufer fallen
dann auf Stimmabdrücke zurück. Schnittstelle wie bisher: `POST /speaker/diarize`.
"""

from __future__ import annotations

from typing import Any

from fastapi import File, Form, HTTPException, UploadFile

from .stimmdaten import RATE, mono_16k
from .unterbau import Einstellungen, Rechner, ZugangFehlt, anwendung, datei_annehmen, wegraeumen

PORT = 8785
VORGABE = "pyannote/speaker-diarization-3.1"


class Motor:
    def __init__(self, modell: str, zugang: str | None) -> None:
        self.modell, self.zugang = modell, zugang
        self.m: Any = None
        self.gescheitert_fuer: str | None = None       # kein erneuter Versuch mit demselben Zugang

    def laden(self, zugang: str | None = None) -> Any:
        zugang = zugang or self.zugang
        schluessel = zugang or "<anmeldung>"
        if self.gescheitert_fuer == schluessel:
            raise ZugangFehlt("Modell mit diesem Zugang nicht ladbar; Zugang prüfen und Dienst neu starten")
        from pyannote.audio import Pipeline
        try:
            try:
                pipe = Pipeline.from_pretrained(self.modell, token=zugang or None)
            except TypeError:                            # ältere pyannote kennen nur use_auth_token
                pipe = Pipeline.from_pretrained(self.modell, use_auth_token=zugang or None)
            if pipe is None:
                raise ZugangFehlt("Zugang fehlt oder die Bedingungen von pyannote/speaker-diarization-3.1 und "
                                  "pyannote/segmentation-3.0 sind nicht angenommen")
        except ZugangFehlt:
            self.gescheitert_fuer = schluessel
            raise
        except Exception as f:
            self.gescheitert_fuer = schluessel
            raise ZugangFehlt(f"Modell nicht ladbar: {type(f).__name__}: {f}"[:300]) from f
        try:
            import torch
            if torch.backends.mps.is_available():
                pipe.to(torch.device("mps"))
        except Exception:
            pass                                          # Prozessor ist die sichere Vorgabe
        self.m, self.gescheitert_fuer = pipe, None
        return pipe

    def trennen(self, pfad: str, anzahl: int | None) -> list[dict]:
        import torch
        wav = mono_16k(pfad)
        kwargs = {"num_speakers": int(anzahl)} if anzahl else {}
        ergebnis = self.m({"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": RATE}, **kwargs)
        annotation = getattr(ergebnis, "speaker_diarization", ergebnis)     # pyannote 4 verpackt, 3 nicht
        return [{"start": round(float(t.start), 3), "end": round(float(t.end), 3), "speaker": str(name)}
                for t, _, name in annotation.itertracks(yield_label=True)]


def erstellen(einstellungen: Einstellungen | None = None, motor: Motor | None = None):
    e = einstellungen or Einstellungen()
    modell = e.modell or VORGABE
    motor = motor or Motor(modell, e.hf_token)
    rechner = Rechner("trennen", modell, e)
    app = anwendung("trennen", rechner, motor.laden, "Sprechertrennung")

    @app.post("/speaker/diarize")
    async def trennen(file: UploadFile = File(...), hf_token: str = Form(""), num_speakers: int | None = Form(None)):
        if rechner.zustand == "ohne_zugang" and hf_token.strip():
            await rechner.laden(lambda: motor.laden(hf_token.strip()))
        if rechner.zustand != "bereit":
            raise HTTPException(503, rechner.meldung or f"Modell {modell} ist nicht bereit ({rechner.zustand})")
        pfad = await datei_annehmen(file, e)
        try:
            return {"turns": await rechner.rechnen(motor.trennen, str(pfad), num_speakers), "model": modell}
        finally:
            wegraeumen(pfad)

    return app
