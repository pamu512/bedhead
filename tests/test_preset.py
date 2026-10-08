"""Preset scaling, clamping, and JSON persistence."""

import json
from dataclasses import asdict, fields

import pytest

from bedhead.config import Preset

STRENGTH_FIELDS = [
    "intensity",
    "skin",
    "under_eye",
    "shine",
    "teeth",
    "hairline",
    "soft_light",
]


class TestScaled:
    @pytest.mark.parametrize("name", STRENGTH_FIELDS)
    @pytest.mark.parametrize("value", [-1.5, -0.01, 0.0, 0.37, 1.0, 1.0001, 7.0])
    def test_strength_is_clamped_to_unit_range(self, name, value):
        preset = Preset(**{name: value})
        assert 0.0 <= preset.scaled(name) <= 1.0

    @pytest.mark.parametrize("name", STRENGTH_FIELDS)
    def test_strength_above_one_saturates(self, name):
        kwargs = {name: 4.2}
        if name != "intensity":
            kwargs["intensity"] = 1.0
        assert Preset(**kwargs).scaled(name) == pytest.approx(1.0)

    @pytest.mark.parametrize("name", STRENGTH_FIELDS)
    def test_negative_strength_gates_to_zero(self, name):
        assert Preset(**{name: -0.3}).scaled(name) == 0.0

    def test_multiplies_by_clamped_intensity(self):
        assert Preset(skin=0.5, intensity=0.8).scaled("skin") == pytest.approx(0.4)

    @pytest.mark.parametrize("intensity", [2.5, 1.0])
    def test_intensity_above_one_clamps_to_one(self, intensity):
        assert Preset(skin=0.5, intensity=intensity).scaled("skin") == pytest.approx(0.5)

    @pytest.mark.parametrize("intensity", [-2.0, 0.0])
    def test_non_positive_intensity_gates_every_strength(self, intensity):
        preset = Preset(intensity=intensity, skin=0.9, shine=0.9)
        assert preset.scaled("skin") == 0.0
        assert preset.scaled("shine") == 0.0


class TestSaveLoad:
    def test_round_trip_preserves_every_field(self, tmp_path):
        preset = Preset(
            intensity=0.9,
            skin=0.8,
            under_eye=0.7,
            shine=0.6,
            teeth=0.5,
            hairline=0.4,
            soft_light=0.3,
            show_original=True,
        )
        path = tmp_path / "preset.json"
        preset.save(str(path))

        loaded = Preset.load(str(path))

        for field in fields(Preset):
            assert getattr(loaded, field.name) == getattr(preset, field.name), field.name
        assert loaded == preset

    def test_round_trip_of_defaults(self, tmp_path):
        path = tmp_path / "preset.json"
        Preset().save(str(path))
        assert Preset.load(str(path)) == Preset()

    def test_unknown_keys_are_dropped(self, tmp_path):
        path = tmp_path / "preset.json"
        payload = asdict(Preset(skin=0.42))
        payload["not_a_field"] = 123
        payload["also_bogus"] = "x"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert Preset.load(str(path)) == Preset(skin=0.42)

    def test_missing_keys_keep_defaults(self, tmp_path):
        path = tmp_path / "preset.json"
        path.write_text(json.dumps({"skin": 0.25}), encoding="utf-8")

        loaded = Preset.load(str(path))

        assert loaded == Preset(skin=0.25)
        defaults = Preset()
        assert loaded.intensity == defaults.intensity
        assert loaded.show_original == defaults.show_original

    def test_non_object_json_is_rejected(self, tmp_path):
        path = tmp_path / "preset.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError):
            Preset.load(str(path))

    def test_non_finite_number_is_rejected(self, tmp_path):
        path = tmp_path / "preset.json"
        path.write_text('{"skin": NaN}', encoding="utf-8")
        with pytest.raises(ValueError):
            Preset.load(str(path))

    def test_non_boolean_show_original_is_rejected(self, tmp_path):
        path = tmp_path / "preset.json"
        path.write_text('{"show_original": 1}', encoding="utf-8")
        with pytest.raises(ValueError):
            Preset.load(str(path))

    def test_save_replaces_atomically(self, tmp_path):
        path = tmp_path / "preset.json"
        Preset(skin=0.2).save(str(path))
        assert not (tmp_path / "preset.json.part").exists()
        assert Preset.load(str(path)).skin == pytest.approx(0.2)


class TestNonFinite:
    def test_nan_strength_clamps_to_zero(self):
        assert Preset(skin=float("nan"), intensity=1.0).scaled("skin") == 0.0

    def test_nan_intensity_gates_every_strength(self):
        preset = Preset(intensity=float("nan"), skin=0.9, shine=0.4)
        assert preset.scaled("skin") == 0.0
        assert preset.scaled("shine") == 0.0
        assert preset.scaled("intensity") == 0.0

    def test_positive_infinity_clamps_to_one(self):
        assert Preset(skin=0.5, intensity=float("inf")).scaled("skin") == pytest.approx(0.5)
        assert Preset(skin=float("inf"), intensity=1.0).scaled("skin") == pytest.approx(1.0)
