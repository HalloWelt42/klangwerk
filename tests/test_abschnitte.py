"""Abschnitte für Hördienste: Schnitt nur in Sprechpausen, längste Pause im letzten Drittel, Notschnitt nur ohne jede
Pause und gekennzeichnet, ohne Sprachpausenerkennung kein Rückfall."""

from __future__ import annotations

import numpy as np
import pytest

from klangwerk.abschnitte import Abschnitt, Sprachpausen, SprachpausenFehlen, pausen_aus, planen, sprachbeginn_vor


def sprache_mit_pausen(pausen: list[tuple[float, float]], ende: float) -> list[tuple[float, float]]:
    """Sprachstellen von 0 bis `ende`, unterbrochen von den Pausen (start, ende)."""
    aus, von = [], 0.0
    for a, b in pausen:
        aus.append((von, a))
        von = b
    aus.append((von, ende))
    return aus


def test_ohne_sprache_keine_abschnitte():
    assert planen([], 100.0, 60) == []


def test_kurz_genug_ein_abschnitt_mit_polster():
    assert planen([(1.0, 20.0)], 30.0, 60) == [Abschnitt(0.9, 20.1)]


def test_schnitt_in_der_laengsten_pause_im_letzten_drittel():
    # Pausen bei 10 s (lang, aber früh), 45 s (0,3 s) und 52 s (0,6 s); Grenze 60 s, letztes Drittel ab 40 s
    sprache = sprache_mit_pausen([(10.0, 11.5), (45.0, 45.3), (52.0, 52.6)], 100.0)
    plan = planen(sprache, 100.0, 60)
    assert plan[0].ende == pytest.approx(52.1) and not plan[0].notschnitt      # 0,1 s hinter der letzten Sprache
    assert plan[1].start == pytest.approx(52.5)                                # 0,1 s vor der nächsten Sprache
    assert all(a.laenge <= 60 for a in plan)
    for a in plan[:-1]:                                                        # jedes Ende liegt in einer Pause
        assert any(p.start <= a.ende <= p.ende for p in pausen_aus(sprache))


def test_kein_schnitt_an_fester_zeit_sondern_in_frueherer_pause():
    # im letzten Drittel keine Pause: dann die längste im ganzen Abschnitt
    sprache = sprache_mit_pausen([(12.0, 12.4), (25.0, 25.9)], 70.0)
    plan = planen(sprache, 70.0, 60)
    assert plan[0].ende == pytest.approx(25.1) and not plan[0].notschnitt


def test_notschnitt_nur_ohne_jede_pause_und_gekennzeichnet():
    plan = planen([(0.0, 130.0)], 130.0, 60)
    assert [a.notschnitt for a in plan] == [True, True, False]
    assert plan[0].ende == pytest.approx(60.0) and plan[1].start == pytest.approx(60.0)


def test_lange_stille_wird_nicht_mitgehoert():
    sprache = [(0.0, 20.0), (200.0, 230.0)]
    plan = planen(sprache, 240.0, 60)
    assert [(round(a.start, 1), round(a.ende, 1)) for a in plan] == [(0.0, 20.1), (199.9, 230.1)]


def test_vorlauf_beginnt_in_einer_pause():
    sprache = [(0.0, 5.0), (5.4, 9.0), (9.5, 12.0)]
    assert sprachbeginn_vor(sprache, 6.2) == pytest.approx(5.3)
    assert sprachbeginn_vor(sprache, 0.0) == 0.0
    assert sprachbeginn_vor(sprache, 8.0) == pytest.approx(7.9)          # Sprachstelle begann viel früher: ab dem Beitrag


def test_ohne_sprachpausenerkennung_kein_rueckfall():
    with pytest.raises(SprachpausenFehlen):
        Sprachpausen().stellen(np.zeros(16000, dtype=np.float32))
