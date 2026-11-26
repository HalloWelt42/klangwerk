"""Modellprüfung ohne Laden (MR-11): gegen einen Wegwerf-Zwischenspeicher, nie den echten."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from huggingface_hub import constants

from klangwerk import modellpruefung, trennen
from klangwerk.unterbau import Einstellungen, ZugangFehlt

KLANGWERK = Path(__file__).resolve().parent.parent


def repo(cache: Path, name: str, dateien: dict[str, bytes]) -> Path:
    """Ein Repo im Aufbau des Zwischenspeichers: refs/main, Schnappschuss mit relativen Links auf blobs."""
    ordner = cache / f"models--{name.replace('/', '--')}"
    rev = "0" * 40
    (ordner / "refs").mkdir(parents=True)
    (ordner / "refs" / "main").write_text(rev)
    for rel, inhalt in dateien.items():
        blob = ordner / "blobs" / f"b{abs(hash(rel))}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(inhalt)
        link = ordner / "snapshots" / rev / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, link.parent), link)
    return ordner


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    c = tmp_path / "huggingface"
    c.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(c))
    monkeypatch.setenv("HF_HUB_CACHE", str(c))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return c


def test_vorhanden_fehlt_und_links_ins_leere(cache: Path):
    ordner = repo(cache, "org/da", {"config.json": b"{}", "weights.safetensors": b"x"})
    assert modellpruefung.fehlt(["org/da"]) is None
    assert "org/weg fehlt im Zwischenspeicher" in modellpruefung.fehlt(["org/da", "org/weg"])
    blob = next((ordner / "blobs").iterdir())
    blob.unlink()
    os.symlink("/Volumes/ki-modelle-nicht-eingehaengt/x/weights.safetensors", blob)
    grund = modellpruefung.fehlt(["org/da"])
    assert grund and "Links ins Leere" in grund and "Hauptlager" in grund


def test_sprechertrennung_nennt_ihre_teilmodelle(cache: Path):
    repo(cache, "pyannote/speaker-diarization-3.1", {"config.yaml": b"pipeline:\n  params:\n    embedding: pyannote/wespeaker\n"
                                                                     b"    segmentation: pyannote/segmentation-3.0\n    clustering: Agglo\n"})
    e = Einstellungen()
    assert trennen.modelle(e) == ["pyannote/speaker-diarization-3.1", "pyannote/wespeaker", "pyannote/segmentation-3.0"]


def test_sprechertrennung_ohne_modell_ist_gestoert_nicht_ohne_zugang(cache: Path):
    m = trennen.Motor("pyannote/speaker-diarization-3.1", None)
    with pytest.raises(RuntimeError, match="Modell fehlt") as f:
        m.laden()
    assert not isinstance(f.value, ZugangFehlt) and m.gescheitert_fuer is None


def test_pruefen_auf_der_befehlszeile(cache: Path):
    def lauf(dienst: str) -> tuple[int, str]:
        r = subprocess.run([sys.executable, "-m", "klangwerk", dienst, "--pruefen"], cwd=KLANGWERK, capture_output=True, text=True,
                           env={**os.environ, "KLANGWERK_MODELL": "org/abdruck"}, timeout=60)
        return r.returncode, r.stdout.strip().splitlines()[-1]
    code, zeile = lauf("sprecher")
    assert code == 1 and zeile.startswith("fehlt: org/abdruck")
    repo(cache, "org/abdruck", {"embedding_model.ckpt": b"x"})
    assert lauf("sprecher") == (0, "vorhanden: org/abdruck")
    assert lauf("ton") == (0, "vorhanden: kein Modell nötig")
