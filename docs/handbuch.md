---
zweck: "Die Sprachdienste des Hauses, je Modell ein eigener Prozess: Sprechen, Stimmklon, Hören, Sprechermerkmal, Sprechertrennung und die Tonarbeit der Apps."
geraet: [mac]
abhaengigkeiten: [maschinenraum]
daten:
  - "Mac: die Modelle in der Modellablage `~/Maschinenraum/modelle/huggingface` (Qwen3-TTS, Whisper Large V3, ECAPA, pyannote)"
  - "Mac: Dienstdateien `~/Maschinenraum/dienste/klangwerk-*.toml` (je Dienst eine)"
  - "Mac: erzeugte Audiodateien bis zum Abholen im Ordner für Zwischendateien des Systems (`klangwerk/`)"
sicherung:
  - "nichts Eigenes zu sichern: der Code liegt im Repo, die Dienstdateien schreibt die Installation neu, die Modelle lassen sich neu laden; die Modellablage gehört zur Sicherung des Macs (siehe Grundlage Mac)"
schluessel_orte:
  - "Mac: `KLANGWERK_HF_TOKEN` unter `[umgebung]` in `~/Maschinenraum/dienste/klangwerk-trennen.toml` (Zugang für das gesperrte Modell der Sprechertrennung)"
einrichten: einrichten.sh
stand: 9efbc13
---

# Klangwerk

## Zweck

Das Klangwerk ersetzt den früheren Sammelprozess für Sprache: jeder Dienst ist ein eigener Prozess mit genau einem
Modell, lädt es beim Start und gibt den Speicher beim Anhalten wieder frei. So schaltet und misst der
[[Maschinenraum]] jeden einzeln, und die [[Leitstelle]] vermittelt die Fähigkeiten an die Apps.

| Dienst | Fähigkeit | Modell |
|---|---|---|
| sprechen | stimme | Qwen3-TTS CustomVoice 8 Bit (neun Stimmen) |
| klonen | stimme_klonen | Qwen3-TTS Base bf16 |
| hoeren | hoeren | Whisper Large V3 (MLX), dazu Silero für Sprechpausen; bis zwei Instanzen |
| sprecher | sprecher_erkennen | ECAPA, auf dem Prozessor |
| trennen | sprecher_trennen | pyannote 3.1 (braucht einen Zugang) |
| ton | ton | ohne Modell: ffmpeg, die Tonarbeit der Apps (wandeln, messen, blenden, legen) |

Die Ports stehen unten im Block "Auf einen Blick" (aus den Dienstdateien). Ohne Klangwerk sprechen und hören
[[Vertonung]] und [[Transkription]] nicht, und die Apps können keinen Ton bearbeiten.

## Was sie braucht

- Mac nach [[Grundlage Mac]]: Python ab 3.11, ffmpeg (Homebrew), die Modellablage.
- [[Maschinenraum]]: die Installation trägt die Dienste in `~/Maschinenraum/dienste/` ein; gestartet werden sie dort
  oder über die Leitstelle, nie von der Installation selbst.
- Die Pakete kommen mit festen Ständen aus `einschraenkungen.txt` (MLX, torch, der Appkern für die Tonarbeit).
- Für die Sprechertrennung ein Zugang bei Hugging Face mit freigeschaltetem pyannote-Modell.

## Neu einrichten

Zwei Wege. Aus der Sicherung: die Modellablage zurückspielen (siehe [[Grundlage Mac]]) und die Dienstdateien mit dem
Ordner `~/Maschinenraum`; dann laufen die Dienste nach der Installation sofort. Frisch: die Installation schreibt die
Dienstdateien neu, die Modelle werden einmal geladen, und die Sprechertrennung bekommt ihren Zugang neu.

### 1. Klon und Installation

```
git clone {{repo}} {{ordner}}
cd {{ordner}} && ./einrichten.sh
```

`./einrichten.sh` ruft `./installieren.sh`: `.venv` im Klon mit den Ständen aus `einschraenkungen.txt`, dann je Dienst
eine Datei `~/Maschinenraum/dienste/klangwerk-<dienst>.toml`, aber nur, wenn es sie noch nicht gibt (Änderungen von
Hand bleiben). Gestartet wird nichts.

Probe: `ls ~/Maschinenraum/dienste/klangwerk-*.toml | wc -l` meldet 6, und der Maschinenraum zeigt die sechs Dienste
unter Dienste.

### 2. Modelle laden

Aus der Sicherung liegen sie in der Modellablage. Frisch einmal je Modelldienst ohne die Sperre fürs Netz starten, so
dass er sein Modell in die Modellablage lädt, und wieder beenden, etwa
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/python -m klangwerk sprechen --port 18781`; die Dienstdateien
setzen danach `HF_HUB_OFFLINE=1`. Für `trennen` vorher den Zugang als `KLANGWERK_HF_TOKEN` unter `[umgebung]` in seine
Dienstdatei schreiben.

Probe: der Dienst meldet auf `/gesundheit` den Zustand `bereit` (beim Laden 503 mit `laedt`, ohne Zugang `ohne_zugang`).

### 3. Starten über den Maschinenraum

Im [[Maschinenraum]] die Dienste starten oder die [[Leitstelle]] sie nach Bedarf starten lassen.

Probe: `curl -s {{pruefadresse:klangwerk-ton}}` meldet `"zustand":"bereit"`, ebenso die Prüfadressen der übrigen Dienste
nach dem Laden ihres Modells; die Leitstelle zeigt die Fähigkeiten unter Dienste.

## Sicherung und Rückspielen

Das Klangwerk hält keine eigenen Daten: der Code liegt im Repo, die Dienstdateien schreibt die Installation neu, und die
erzeugten Audiodateien leben nur bis zum Abholen. Was eine Neueinrichtung dauern lässt, sind die Modelle in der
Modellablage (mehrere Gigabyte); sie gehören zur Sicherung des Macs (siehe [[Grundlage Mac]]). Der Zugang der
Sprechertrennung steht nur in ihrer Dienstdatei und gehört mit `~/Maschinenraum` in dieselbe Sicherung.

## Wenn etwas klemmt

- Ein Dienst bleibt bei `laedt` oder endet sofort: das Modell fehlt in der Modellablage und die Dienstdatei sperrt das
  Netz (`HF_HUB_OFFLINE=1`); Schritt 2.
- `trennen` meldet `ohne_zugang`: `KLANGWERK_HF_TOKEN` fehlt oder das Modell ist für den Zugang nicht freigeschaltet.
- Der Tondienst meldet nicht bereit: ffmpeg oder ffprobe fehlen im Suchpfad des Maschinenraums (Homebrew).
- Speicher knapp: je Dienst ein Prozess mit seinem Modell (zusammen rund 16 GB); die Leitstelle hält Dienste an, die
  sie nicht braucht.
