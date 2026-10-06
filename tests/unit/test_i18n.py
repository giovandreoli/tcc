from __future__ import annotations

import json

import pytest

from spectra.i18n import Catalog, load_catalog, set_locale, t


def test_catalog_flattens_nested_keys():
    catalog = load_catalog("pt_BR")
    assert "mode.menu" in catalog
    assert "physio.exercise.open_close" in catalog
    assert catalog.t("mode.menu") == "Menu Principal"


def test_unknown_key_is_returned_verbatim():
    catalog = load_catalog("pt_BR")
    assert catalog.t("does.not.exist") == "does.not.exist"


def test_placeholders_are_formatted():
    catalog = load_catalog("pt_BR")
    assert catalog.t("common.score", score=3, total=5) == "Placar: 3/5"


def test_missing_placeholder_falls_back_to_template():
    catalog = Catalog("xx", {"greet": "Olá {name}"})
    assert catalog.t("greet", other=1) == "Olá {name}"


def test_unknown_locale_falls_back_to_default(tmp_path):
    (tmp_path / "pt_BR.json").write_text(json.dumps({"a": "b"}), encoding="utf-8")
    catalog = load_catalog("de_DE", locales_dir=tmp_path)
    assert catalog.locale == "pt_BR"


def test_module_level_translator_uses_active_catalog():
    set_locale("pt_BR")
    assert t("common.back") == "Voltar"


@pytest.mark.parametrize(
    "key",
    [
        "app.title",
        "common.quit",
        "mode.free_draw",
        "paint.saved",
        "edu_count.correct",
        "physio.instructions.finger_wave_3",
    ],
)
def test_required_keys_exist(key):
    assert key in load_catalog("pt_BR")
