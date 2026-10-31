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
"$HIER/.venv/bin/pip" install --quiet -c "$HIER/einschraenkungen.txt" -e "$HIER[mlx,torch,ton]"
echo "Umgebung bereit: $HIER/.venv"
# Hören mit MOSS braucht neuere transformers als die übrigen Dienste: eigene Umgebung
[ -d "$HIER/.venv-moss" ] || "$PYTHON" -m venv "$HIER/.venv-moss"
"$HIER/.venv-moss/bin/pip" install --quiet --upgrade pip
"$HIER/.venv-moss/bin/pip" install --quiet -c "$HIER/einschraenkungen-moss.txt" -e "$HIER[moss,test]"
echo "Umgebung bereit: $HIER/.venv-moss"

# Varianten: [modul] Python-Modul des Dienstes (Vorgabe die Kennung), [umgebung] eine weitere Zeile unter [umgebung],
# [venv] Umgebung (Vorgabe .venv)
dienstdatei() {   # kennung name beschreibung port faehigkeit speicher_gb [startzeit_s] [plaetze] [instanzen] [modul] [umgebung] [venv]
  datei="$DIENSTE/klangwerk-$1.toml"
  if [ -f "$datei" ]; then echo "bleibt: $datei"; return; fi
  cat > "$datei" <<EOF
# Klangwerk: $2. Je Prozess genau ein Modell (der Tondienst ohne); es lädt beim Start, laufen heißt bereit.
name = "Klangwerk $2"
beschreibung = "$3"
ordner = "$HIER"
befehl = ["${12:-.venv}/bin/python", "-m", "klangwerk", "${10:-$1}", "--port", "\$MASCHINENRAUM_PORT"]
port = $4
pruefadresse = "http://127.0.0.1:$4/gesundheit"
faehigkeiten = ["$5"]
speicher_gb = $6
startzeit_s = ${7:-300}
anhaltezeit_s = 30
plaetze = ${8:-1}
instanzen = ${9:-1}

[umgebung]
HF_HUB_OFFLINE = "1"
${11:-}
EOF
  echo "eingetragen: $datei"
}

if [ -d "$DIENSTE" ]; then
  dienstdatei sprechen "Sprechen" "Sprachsynthese mit Qwen3-TTS CustomVoice (9 Stimmen)" 8781 stimme 3
  dienstdatei klonen "Stimmklon" "Stimme aus einer Referenz klonen, Qwen3-TTS Base" 8782 stimme_klonen 6
  dienstdatei hoeren "Hören" "Spracherkennung mit Whisper Large V3" 8783 hoeren 4 300 1 2
  "$HIER/.venv/bin/python" -m klangwerk.selbstauskunft hoeren "$DIENSTE/klangwerk-hoeren.toml"   # Sprachen des Modells
  # Varianten des Hörens: weitere Dienste mit eigenem Modell, für die Apps je Auftrag wählbar
  dienstdatei hoeren-8bit "Hören 8 Bit" "Spracherkennung mit Whisper Large V3 in 8 Bit" 8788 hoeren 3 300 1 1 \
    hoeren 'KLANGWERK_MODELL = "mlx-community/whisper-large-v3-mlx-8bit"'
  "$HIER/.venv/bin/python" -m klangwerk.selbstauskunft hoeren "$DIENSTE/klangwerk-hoeren-8bit.toml"
  dienstdatei hoeren-turbo "Hören Turbo" "Spracherkennung mit Whisper Large V3 Turbo (schneller, kleinerer Dekodierer)" 8789 hoeren 3 300 1 1 \
    hoeren 'KLANGWERK_MODELL = "mlx-community/whisper-large-v3-turbo"'
  "$HIER/.venv/bin/python" -m klangwerk.selbstauskunft hoeren "$DIENSTE/klangwerk-hoeren-turbo.toml"
  # Sprachen nennt die Herkunft nicht einzeln (über 50); ohne Liste prüft die App nur die Form
  dienstdatei hoeren-moss "Hören mit Sprechern" "Spracherkennung mit Sprechertrennung und Zeitmarken in einem Durchgang (MOSS-Transcribe-Diarize 0.9B)" 8787 hoeren 6 300 1 1 \
    hoeren-moss 'PYTORCH_ENABLE_MPS_FALLBACK = "1"' .venv-moss
  dienstdatei sprecher "Sprechermerkmal" "Stimmabdruck je Abschnitt (ECAPA), auf dem Prozessor" 8784 sprecher_erkennen 1
  # Sprechertrennung braucht KLANGWERK_HF_TOKEN unter [umgebung] ihrer Dienstdatei
  dienstdatei trennen "Sprechertrennung" "Wer spricht wann: trennt die Sprecher einer Aufnahme nach Zeit (pyannote)" 8785 sprecher_trennen 2
  dienstdatei ton "Ton" "Tonarbeit der Apps ohne Modell: wandeln, messen, blenden, legen (ffmpeg)" 8786 ton 1 30 4
else
  echo "Kein Maschinenraum unter $DIENSTE; Dienste von Hand starten: $HIER/.venv/bin/klangwerk <dienst>"
fi
