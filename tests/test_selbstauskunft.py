"""Selbstauskunft: die Sprachen kommen einmal hinter die Fähigkeiten der Dienstdatei, eine vorhandene Angabe bleibt."""

from klangwerk.selbstauskunft import eintragen


def test_sprachen_einmal_eintragen(tmp_path):
    datei = tmp_path / "klangwerk-hoeren.toml"
    datei.write_text('name = "Klangwerk Hören"\nfaehigkeiten = ["hoeren"]\nport = 8783\n')
    assert eintragen(datei, ["de", "ru"])
    assert datei.read_text().splitlines()[2] == 'sprachen = ["de", "ru"]'
    assert not eintragen(datei, ["en"])
    assert datei.read_text().count("sprachen") == 1
