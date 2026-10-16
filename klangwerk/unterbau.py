"""Gemeinsamer Unterbau aller Klangwerk-Dienste.

Jeder Dienst ist ein eigener Prozess mit genau einem Modell. Er lädt es beim Start; solange es lädt, antwortet
`/gesundheit` mit 503, danach mit 200. Laufen heißt damit: das Modell liegt im Speicher. Ein Maschinenraum misst
den Prozess und kann ihn starten und anhalten; eine Leitstelle plant mit seinem Bedarf.

Grafik: Innerhalb eines Prozesses läuft immer nur eine Rechnung zugleich (eine Sperre); mehrere Rechnungen in
einem Prozess brachten früher die Metal-Befehlspuffer zum Absturz. Getrennte Prozesse laufen gleichzeitig stabil
(gemessen am 16.10.2026: Synthese und Erkennung je 30 und 40 Durchläufe parallel, ohne Fehler, je etwa ein Drittel
langsamer). Der Zwischenspeicher für Grafikpuffer von MLX wächst ohne Grenze; er wird begrenzt und nach jeder
Rechnung geleert, die Modelle bleiben davon unberührt.

Feste Zeitgrenzen für Arbeit gibt es nicht: Abbrechen ist Sache des Aufrufers oder der Leitstelle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import traceback
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import __version__

log = logging.getLogger("klangwerk")
AUDIO_ENDUNGEN = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm")

# ffmpeg und ffprobe liegen auf dem Mac unter Homebrew
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")


class Einstellungen(BaseSettings):
    """Je Dienst über Umgebungsvariablen mit der Vorsilbe KLANGWERK_ (Dienstdatei im Maschinenraum)."""

    model_config = SettingsConfigDict(env_prefix="KLANGWERK_", env_file=None)

    adresse: str = Field(default="127.0.0.1", description="nur dieser Rechner; erreichbar ist der Dienst über die Leitstelle")
    modell: str | None = Field(default=None, description="Modell dieses Dienstes; leer: Vorgabe des Dienstes")
    mlx_zwischenspeicher_gb: float = Field(default=2.0, ge=0, description="Grenze für behaltene Grafikpuffer von MLX")
    hochladen_gb: float = Field(default=2.0, gt=0, description="größte angenommene Audiodatei")
    ergebnisse: Path = Field(default=Path(tempfile.gettempdir()) / "klangwerk", description="Ordner für erzeugte Audiodateien")
    hf_token: str | None = Field(default=None, description="Zugang für gesperrte Modelle (Sprechertrennung)")


class Gesundheit(BaseModel):
    dienst: str
    fassung: str
    modell: str
    zustand: str                      # laedt, bereit, gestoert, ohne_zugang
    meldung: str = ""


class Rechner:
    """Hält das eine Modell des Dienstes und die Sperre, unter der gerechnet wird."""

    def __init__(self, dienst: str, modell: str, einstellungen: Einstellungen) -> None:
        self.dienst, self.modell, self.einstellungen = dienst, modell, einstellungen
        self.zustand, self.meldung = "laedt", ""
        self.geladen: Any = None
        self._sperre = asyncio.Lock()
        self._mlx_begrenzt = False

    def _mlx_begrenzen(self) -> None:
        if self._mlx_begrenzt:
            return
        self._mlx_begrenzt = True
        try:
            import mlx.core as mx
            mx.set_cache_limit(int(self.einstellungen.mlx_zwischenspeicher_gb * 1024**3))
        except Exception as f:                 # ohne MLX (Dienste auf dem Prozessor) gibt es nichts zu begrenzen
            log.debug("MLX-Zwischenspeicher nicht begrenzbar: %s", f)

    @staticmethod
    def _mlx_leeren() -> None:
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    async def rechnen(self, funktion: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Eine Rechnung zugleich, in einem Thread; danach behaltene Grafikpuffer freigeben."""
        if self.zustand != "bereit":
            raise HTTPException(503, f"Modell {self.modell} ist nicht bereit ({self.zustand}){': ' + self.meldung if self.meldung else ''}")
        self._mlx_begrenzen()
        async with self._sperre:
            try:
                return await asyncio.to_thread(funktion, *args, **kwargs)
            finally:
                self._mlx_leeren()

    async def laden(self, lader: Callable[[], Any]) -> None:
        self._mlx_begrenzen()
        try:
            self.geladen = await asyncio.to_thread(lader)
            self.zustand, self.meldung = "bereit", ""
            log.info("%s: Modell %s geladen", self.dienst, self.modell)
        except ZugangFehlt as f:
            self.zustand, self.meldung = "ohne_zugang", str(f)
            log.warning("%s: %s", self.dienst, f)
        except Exception as f:
            self.zustand, self.meldung = "gestoert", f"{type(f).__name__}: {f}"[:300]
            log.error("%s: Modell %s nicht ladbar\n%s", self.dienst, self.modell, traceback.format_exc())


class ZugangFehlt(RuntimeError):
    """Das Modell ist gesperrt und es fehlt ein Zugang; der Dienst läuft, kann aber erst mit Zugang rechnen."""


async def datei_annehmen(datei: UploadFile, einstellungen: Einstellungen, leer_meldung: str = "Leere Audio-Datei") -> Path:
    """Hochgeladene Audiodatei in eine Zwischendatei mit passender Endung (sonst scheitern Decoder an MP3 in .wav)."""
    endung = Path(datei.filename or "audio.wav").suffix.lower() or ".wav"
    if endung not in AUDIO_ENDUNGEN:
        endung = ".wav"
    inhalt = await datei.read()
    if not inhalt:
        raise HTTPException(400, leer_meldung)
    grenze = int(einstellungen.hochladen_gb * 1024**3)
    if len(inhalt) > grenze:
        raise HTTPException(413, f"Audio zu groß (höchstens {grenze // 1024**2} MB)")
    fd, name = tempfile.mkstemp(suffix=endung, prefix="klangwerk-")
    with os.fdopen(fd, "wb") as f:
        f.write(inhalt)
    return Path(name)


def wegraeumen(*pfade: Path | str | None) -> None:
    for p in pfade:
        if p:
            try:
                os.unlink(p)
            except OSError:
                pass


def anwendung(dienst: str, rechner: Rechner, lader: Callable[[], Any], titel: str) -> FastAPI:
    """FastAPI-Anwendung eines Dienstes: lädt das Modell beim Start im Hintergrund, Gesundheit, Fehler als JSON."""
    rechner.einstellungen.ergebnisse.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lebenslauf(_app: FastAPI):
        aufgabe = asyncio.create_task(rechner.laden(lader))
        try:
            yield
        finally:
            aufgabe.cancel()

    app = FastAPI(title=f"Klangwerk {titel}", version=__version__, lifespan=lebenslauf)
    app.state.rechner = rechner

    @app.exception_handler(Exception)
    async def fehler(request: Request, f: Exception):
        log.error("Fehler in %s: %s\n%s", request.url.path, f, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": type(f).__name__, "message": str(f)[:500]})

    def gesundheit() -> JSONResponse:
        g = Gesundheit(dienst=dienst, fassung=__version__, modell=rechner.modell, zustand=rechner.zustand, meldung=rechner.meldung)
        # 503, solange das Modell nicht rechnen kann: der Maschinenraum wartet beim Start darauf
        return JSONResponse(g.model_dump(), status_code=200 if rechner.zustand == "bereit" else 503)

    app.add_api_route("/gesundheit", gesundheit, methods=["GET"])
    app.add_api_route("/health", gesundheit, methods=["GET"], include_in_schema=False)   # wie der bisherige Sprachdienst
    return app


def starten(app: FastAPI, port: int, einstellungen: Einstellungen) -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=einstellungen.adresse, port=port, log_level="warning", access_log=False)


Lader = Callable[[], Any]
Rechnung = Callable[..., Awaitable[Any]]
