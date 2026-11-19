# Reparatur der Sprachausgabe

Reparatur vom 19.11.2026 aus HELF-243, umgesetzt in HELF-244, HELF-245 und HELF-246.

## Sprachwahl und vollständige Ausgabe

Die öffentlichen Schnittstellen behalten `language`. Der gemeinsame Übergang in `klangwerk/synthese.py` bildet den
Wert auf `lang_code` ab, wie ihn die installierte Modellbibliothek erwartet. Eine bewusst gewählte automatische Sprache
geht als `auto` weiter. Der Dienst übernimmt alle gelieferten Audioteile in ihrer Reihenfolge. Leere Teile werden
übersprungen; nur wenn überhaupt kein Ton entsteht, greift die vorhandene Wiederholung.

## Tempo

Die Modellbibliothek erzeugt den Ton in ihrem ursprünglichen Tempo. Anschließend ändert die gemeinsame Tonverarbeitung
aus dem Appkern das Tempo bei gleicher Tonhöhe, genau einmal. Die öffentlichen Werte 0,1 bis 4,0 bleiben unterstützt;
Werte unter 0,5 entstehen durch mehrere zulässige Schritte. Tempo 1,0 lässt den Ton unverändert.

Die Tokenbegrenzung richtet sich ausschließlich nach dem Text. Ein schneller gewähltes Tempo darf die Erzeugung nicht
vorzeitig beenden. Das zusätzliche Stauchen bei zeitgenauer Vertonung bleibt eine eigene Operation: Es passt die bereits
im gewählten Tempo erzeugte Aufnahme an ihr Zeitfenster an.

Die Tempoänderung benötigt die bereits für den Tondienst installierte Abhängigkeit `appkern[tondienst]` und ffmpeg.
Modelle werden weiterhin ausschließlich durch den Maschinenraum und die Leitstelle verwaltet.

## Kurze Satzanfänge

Die Ausgabe wird nicht mehr allein wegen eines kurzen Anfangslauts mit nachfolgender Pause beschnitten. Eine
Lautstärkemessung kann ein ungewolltes Artefakt nicht zuverlässig von einem beabsichtigten ersten Wort unterscheiden.
Ein verbleibender ungewollter Laut muss daher gezielt geprüft werden; der Dienst entfernt keine vermuteten Wörter.

## Prüfung

`tests/test_synthese.py` prüft beide echten Dienstmotoren mit mehreren und leeren Audioteilen. Zusätzlich wird die
installierte Erzeugungsfunktion ohne ihre teure Modellrechnung ausgeführt, damit ein falscher Parametername auffällt.
Reale Tonoperationen prüfen Dauer und Frequenz bei unterschiedlichen Tempi sowie alle drei öffentlichen Sprechwege.
`tests/test_dienste.py` sichert den vollständigen kurzen Sprechbeginn samt Pause ab.

Diese Reparatur garantiert keine korrekte Aussprache jedes Anglizismus und keine bestimmte persönliche Stimmähnlichkeit.
Das wird mit identischen Texten und derselben Stimmprobe in einem eigenen Hörvergleich bewertet.
