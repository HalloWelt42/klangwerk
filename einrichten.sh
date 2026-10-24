#!/usr/bin/env bash
# Einrichten des Klangwerks. Schritte und Proben der Neueinrichtung stehen in docs/handbuch.md.
#
#   ./einrichten.sh                                       wie ./installieren.sh: Umgebung mit den erprobten Paketständen,
#                                                         je Dienst eine Dienstdatei im Maschinenraum (nur wenn sie fehlt)
#   ./einrichten.sh --probe --ziel DIR --port-versatz N   Probelauf des Handbuchs: dieselbe Installation in DIR, die
#                                                         Dienstdateien in einen eigenen Ordner DIR/maschinenraum (nie in
#                                                         den echten), dann der Tondienst (ohne Modell) als Prozess auf
#                                                         seinem Port plus N bis "bereit"; die Modelldienste laden
#                                                         Gigabytes und laufen im Probelauf nicht
#   ./einrichten.sh --probe --ziel DIR --abbauen          beendet den Tondienst
set -euo pipefail
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WARTEN_S=60
DIENSTE_ANZAHL=6

probe_aufbauen() {
  local ziel="$1" versatz="$2" dienste="$1/maschinenraum/dienste" port anzahl
  mkdir -p "$dienste" "$ziel/ergebnisse"
  MASCHINENRAUM_ORDNER="$ziel/maschinenraum" "$ziel/installieren.sh"
  anzahl="$(find "$dienste" -name 'klangwerk-*.toml' | wc -l | tr -d ' ')"
  if [ "$anzahl" != "$DIENSTE_ANZAHL" ]; then
    echo "Die Installation schrieb $anzahl statt $DIENSTE_ANZAHL Dienstdateien" >&2
    return 1
  fi
  # Port des Tondienstes aus seiner Dienstdatei, verschoben um den Versatz
  port=$(( $(sed -n 's/^port = //p' "$dienste/klangwerk-ton.toml") + versatz ))
  KLANGWERK_ERGEBNISSE="$ziel/ergebnisse" nohup "$ziel/.venv/bin/python" -m klangwerk ton --port "$port" \
    > "$ziel/ton.log" 2>&1 &
  echo $! > "$ziel/ton.pid"
  local ende=$(( $(date +%s) + WARTEN_S ))
  while [ "$(date +%s)" -lt "$ende" ]; do
    if curl -sf "http://127.0.0.1:$port/gesundheit" | grep -q '"zustand":"bereit"'; then
      echo "ANTWORTET http://127.0.0.1:$port/gesundheit"
      return 0
    fi
    sleep 1
  done
  echo "Der Tondienst ist nach $WARTEN_S s nicht bereit" >&2
  tail -n 40 "$ziel/ton.log" >&2 || true
  return 1
}

probe_abbauen() {
  local pid
  if [ -f "$1/ton.pid" ]; then
    pid="$(cat "$1/ton.pid")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$1/ton.pid"
  fi
}

probe=0; abbauen=0; ziel=""; versatz=""
while [ $# -gt 0 ]; do
  case "$1" in
    --probe) probe=1 ;;
    --abbauen) abbauen=1 ;;
    --ziel) ziel="${2:?--ziel braucht einen Ordner}"; shift ;;
    --port-versatz) versatz="${2:?--port-versatz braucht eine Zahl}"; shift ;;
    *) echo "unbekannte Angabe: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$probe" = 0 ]; then
  exec "$HIER/installieren.sh"
elif [ -z "$ziel" ] || [ ! -d "$ziel" ]; then
  echo "--probe braucht --ziel mit einem vorhandenen Ordner" >&2; exit 2
elif [ "$abbauen" = 1 ]; then
  probe_abbauen "$(cd "$ziel" && pwd)"
else
  probe_aufbauen "$(cd "$ziel" && pwd)" "${versatz:?--port-versatz fehlt}"
fi
