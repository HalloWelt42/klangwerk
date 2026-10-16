#!/bin/sh
# Klangwerk installieren oder aktualisieren: Umgebung mit den erprobten Paketständen, dann je Dienst eine Datei im
# Maschinenraum (~/Maschinenraum/dienste/klangwerk-<dienst>.toml), sofern es den Maschinenraum gibt und die Datei
# noch nicht existiert. Gestartet wird nichts; das geschieht im Maschinenraum oder über die Leitstelle.
set -eu

HIER="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
DIENSTE="${MASCHINENRAUM_ORDNER:-$HOME/Maschinenraum}/dienste"

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || { echo "Python 3.11 oder neuer nötig" >&2; exit 1; }
[ -d "$HIER/.venv" ] || "$PYTHON" -m venv "$HIER/.venv"
"$HIER/.venv/bin/pip" install --quiet --upgrade pip
"$HIER/.venv/bin/pip" install --quiet -c "$HIER/einschraenkungen.txt" -e "$HIER[mlx,torch]"
echo "Umgebung bereit: $HIER/.venv"

dienstdatei() {   # kennung name beschreibung port faehigkeit speicher_gb
  datei="$DIENSTE/klangwerk-$1.toml"
  if [ -f "$datei" ]; then echo "bleibt: $datei"; return; fi
  cat > "$datei" <<EOF
# Klangwerk: $2. Je Prozess genau ein Modell; es lädt beim Start, laufen heißt geladen.
name = "Klangwerk $2"
beschreibung = "$3"
ordner = "$HIER"
befehl = [".venv/bin/python", "-m", "klangwerk", "$1"]
port = $4
pruefadresse = "http://127.0.0.1:$4/gesundheit"
faehigkeiten = ["$5"]
speicher_gb = $6
startzeit_s = 300
anhaltezeit_s = 30

[umgebung]
HF_HUB_OFFLINE = "1"
EOF
  echo "eingetragen: $datei"
}

if [ -d "$DIENSTE" ]; then
  dienstdatei sprechen "Sprechen" "Sprachsynthese mit Qwen3-TTS CustomVoice (9 Stimmen)" 8781 stimme 3
  dienstdatei klonen "Stimmklon" "Stimme aus einer Referenz klonen, Qwen3-TTS Base" 8782 stimme_klonen 6
  dienstdatei hoeren "Hören" "Spracherkennung mit Whisper Large V3" 8783 hoeren 4
  dienstdatei sprecher "Sprechermerkmal" "Stimmabdruck je Abschnitt (ECAPA), auf dem Prozessor" 8784 sprecher_erkennen 1
  dienstdatei trennen "Sprechertrennung" "Wer spricht wann (pyannote); braucht KLANGWERK_HF_TOKEN unter [umgebung]" 8785 sprecher_trennen 2
else
  echo "Kein Maschinenraum unter $DIENSTE; Dienste von Hand starten: $HIER/.venv/bin/klangwerk <dienst>"
fi
