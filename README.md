# Klangwerk

Sprachdienste, je Modell ein eigener Prozess: Sprechen, Stimmklon, Hören, Sprechermerkmal und Sprechertrennung.
Jeder Dienst lädt beim Start genau sein Modell; laufen heißt geladen, anhalten heißt der Speicher ist wieder frei.
So lässt sich jeder einzeln schalten und messen, statt dass ein Sammelprozess alle Modelle auf einmal hält.

## Zusammenspiel

Das Klangwerk gehört zu [Leitstelle](https://github.com/HalloWelt42/leitstelle),
[Maschinenraum](https://github.com/HalloWelt42/maschinenraum) und [Stellwerk](https://github.com/HalloWelt42/stellwerk).
Der Maschinenraum des Rechners startet, hält an und misst die Dienste; die Leitstelle vermittelt die Anfragen der
Anwendungen an sie und plant ihren Speicher ein. Die Dienste lauschen nur auf ihrem Rechner (`127.0.0.1`);
Anwendungen erreichen sie über die Leitstelle. Die Anwendungen selbst, etwa eine Sprachwerkstatt, sind Webapps im
Stellwerk und halten keine eigenen Modelle.

```
Webapps (Stellwerk)  ->  Leitstelle  ->  Maschinenraum  ->  Klangwerk sprechen, klonen, hoeren, sprecher, trennen
```

## Die Dienste

| Dienst | Port | Fähigkeit | Modell (Vorgabe) | Rechnet auf |
|---|---|---|---|---|
| `sprechen` | 8781 | stimme | Qwen3-TTS 1.7B CustomVoice 8-bit | Grafik (MLX) |
| `klonen` | 8782 | stimme_klonen | Qwen3-TTS 1.7B Base bf16 | Grafik (MLX) |
| `hoeren` | 8783 | hoeren | Whisper Large V3, Silero für Sprachpausen | Grafik (MLX) |
| `sprecher` | 8784 | sprecher_erkennen | ECAPA-TDNN (SpeechBrain) | Prozessor |
| `trennen` | 8785 | sprecher_trennen | pyannote speaker-diarization 3.1 | Grafik (torch) |

Ein anderes Modell für denselben Zweck (etwa VoiceDesign statt CustomVoice) ist ein weiterer Dienst mit
`KLANGWERK_MODELL` in seiner Dienstdatei. Je Prozess bleibt es bei einem Modell.

Innerhalb eines Dienstes rechnet immer nur ein Auftrag zugleich. Mehrere Dienste laufen gleichzeitig: gemessen liefen
Synthese und Erkennung als getrennte Prozesse je 30 und 40 Durchläufe parallel ohne Fehler, jeder etwa ein Drittel
langsamer als allein. Der Zwischenspeicher für Grafikpuffer ist begrenzt (`KLANGWERK_MLX_ZWISCHENSPEICHER_GB`, Vorgabe
2) und wird nach jeder Rechnung geleert. Feste Zeitgrenzen für Arbeit gibt es nicht.

## Schnittstellen

Jeder Dienst: `GET /gesundheit` (200, sobald das Modell bereit ist, sonst 503 mit `zustand` laedt, ohne_zugang oder
gestoert).

| Dienst | Aufrufe |
|---|---|
| sprechen | `POST /v1/audio/speech` (übliches Format, Antwort WAV); `POST /tts` (JSON, Datei unter `GET /tts/audio/<datei>`); `GET /tts/voices` |
| klonen | `POST /tts/clone` (Formular: `text`, `ref_audio`, `ref_text`, `language`, `speed`); `GET /tts/audio/<datei>` |
| hoeren | `POST /v1/audio/transcriptions` (`json`, `text`, `verbose_json`); `POST /stt` (Segmente mit Wortzeiten); `POST /stt/stream` (Ereignisstrom je Segment) |
| sprecher | `POST /speaker/embed` (Datei und `segments` als JSON-Liste von start, end; Abdruck mit 192 Werten oder null unter 1 s) |
| trennen | `POST /speaker/diarize` (Datei, optional `num_speakers`, `hf_token`); Antwort `turns` mit start, end, speaker |

Die Aufrufe `/tts`, `/tts/clone`, `/stt`, `/stt/stream` und `/speaker/*` entsprechen einem älteren Sammeldienst, damit
Anwendungen ohne Umbau umziehen können. Übernommen sind auch seine Nachbesserungen: Wiederholung bei leerer Ausgabe,
Entfernen eines kurzen Artefakts am Satzanfang beim Klonen, Schutz vor Halluzinationen und Abschnitte von 6 bis 15 s
mit Wortzeiten beim Hören.

## Installieren

Python 3.11 oder neuer, ffmpeg; die Modelle laden beim ersten Start aus dem Netz oder aus dem Zwischenspeicher.

```bash
./installieren.sh
```

Das Skript legt `.venv` mit den erprobten Paketständen aus `einschraenkungen.txt` an und trägt, wenn es einen
Maschinenraum gibt, je Dienst eine Datei unter `~/Maschinenraum/dienste/` ein (vorhandene bleiben). Die
Sprechertrennung braucht einen Zugang zu den gesperrten pyannote-Modellen: `KLANGWERK_HF_TOKEN` unter `[umgebung]` in
ihrer Dienstdatei oder die Anmeldung von huggingface_hub auf dem Rechner. Von Hand:

```bash
.venv/bin/klangwerk sprechen
```

## Einstellungen

Je Dienst über Umgebungsvariablen (in der Dienstdatei des Maschinenraums unter `[umgebung]`):

| Variable | Vorgabe | Bedeutung |
|---|---|---|
| `KLANGWERK_MODELL` | je Dienst | Modell dieses Dienstes |
| `KLANGWERK_ADRESSE` | 127.0.0.1 | nur dieser Rechner |
| `KLANGWERK_MLX_ZWISCHENSPEICHER_GB` | 2 | Grenze für behaltene Grafikpuffer |
| `KLANGWERK_HOCHLADEN_GB` | 2 | größte angenommene Audiodatei |
| `KLANGWERK_ERGEBNISSE` | Zwischenordner/klangwerk | Ablage erzeugter Audiodateien |
| `KLANGWERK_HF_TOKEN` | keiner | Zugang für die Sprechertrennung |

## Entwickeln

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python -m pytest -m echt
```

Der erste Aufruf prüft alle Dienste mit Schein-Modellen. Der zweite läuft mit den echten Modellen eine Kette: sprechen,
wiedererkennen, Stimme klonen, Stimmabdrücke vergleichen.

---

## Lizenz

**Nicht-kommerzielle Nutzung** - Siehe [LICENSE](LICENSE)

Erlaubt: Private Nutzung, Installation, persönliche Anpassungen, Teilen mit Quellenangabe

Verboten: Kommerzielle Nutzung, Verkauf, Einbindung in kommerzielle Produkte

---

## Unterstützen

Klangwerk ist ein privates Hobby-Projekt. Kein Tracking, keine Werbung, keine Kompromisse.

Wenn dir das Projekt gefällt, kannst du ihm einen Stern geben - oder direkt hier:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/HalloWelt42)

**Crypto:**

| Coin | Adresse |
|------|---------|
| BTC | `bc1qnd599khdkv3v3npmj9ufxzf6h4fzanny2acwqr` |
| DOGE | `DL7tuiYCqm3xQjMDXChdxeQxqUGMACn1ZV` |
| ETH | `0x8A28fc47bFFFA03C8f685fa0836E2dBe1CA14F27` |

Copyright (c) 2026 HalloWelt42
