"""``ensure_models`` downloads each model once and never re-downloads.

All filesystem activity is redirected to ``tmp_path`` via a monkeypatched
``MODEL_DIR``; ``./models/`` is never read or written by this module's tests.
"""

import hashlib
import urllib.error
from pathlib import Path

import bedhead.models as models_mod


def _patch_download(monkeypatch, model_dir, calls, fail_on=None):
    def fake_urlretrieve(url, filename, reporthook=None, *args, **kwargs):
        calls.append(url)
        dest = Path(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"stand-in model bytes")
        if fail_on is not None and len(calls) == fail_on:
            raise urllib.error.URLError("offline")
        return filename, {"Content-Length": "21"}

    # keep the real sha256 verification path active: repin MODELS to the
    # hash of the stand-in bytes so the check exercises and passes
    standin_sha = hashlib.sha256(b"stand-in model bytes").hexdigest()
    patched_models = {
        name: (url, standin_sha) for name, (url, _sha) in models_mod.MODELS.items()
    }
    monkeypatch.setattr(models_mod, "MODELS", patched_models)
    monkeypatch.setattr(models_mod, "MODEL_DIR", model_dir)
    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)


def _assert_under(paths, model_dir):
    assert all(Path(p).is_relative_to(model_dir) for p in paths.values())


def test_first_call_downloads_each_model_once(tmp_path, monkeypatch):
    urls = []
    _patch_download(monkeypatch, tmp_path, urls)

    paths = models_mod.ensure_models()

    assert set(paths) == set(models_mod.MODELS)
    assert all(isinstance(p, Path) and p.exists() and p.stat().st_size > 0 for p in paths.values())
    assert urls == [v[0] for v in models_mod.MODELS.values()]
    _assert_under(paths, tmp_path)
    assert list(tmp_path.glob("*.part")) == []


def test_second_call_is_a_cache_hit(tmp_path, monkeypatch):
    calls = []
    _patch_download(monkeypatch, tmp_path, calls)

    first = models_mod.ensure_models()
    n_first = len(calls)
    second = models_mod.ensure_models()

    assert n_first == len(models_mod.MODELS)
    assert second == first
    assert len(calls) == n_first, "second ensure_models() must not download again"


def test_partial_cache_downloads_only_the_missing_entry(tmp_path, monkeypatch):
    calls = []
    _patch_download(monkeypatch, tmp_path, calls)

    kept = next(iter(models_mod.MODELS))
    (tmp_path / kept).write_bytes(b"already here")
    paths = models_mod.ensure_models()

    assert len(calls) == len(models_mod.MODELS) - 1
    assert models_mod.MODELS[kept][0] not in calls
    _assert_under(paths, tmp_path)


def test_failed_download_leaves_no_partial(tmp_path, monkeypatch):
    calls = []
    nested = tmp_path / "fresh"
    _patch_download(monkeypatch, nested, calls, fail_on=1)

    try:
        models_mod.ensure_models()
    except urllib.error.URLError as exc:
        assert "offline" in str(exc.reason)
    else:
        raise AssertionError("ensure_models() swallowed the download error")

    assert calls == [next(iter(models_mod.MODELS.values()))[0]]
    assert list(nested.glob("*.part")) == []
    assert list(nested.glob("*.task")) == []
    assert list(nested.glob("*.tflite")) == []


def test_second_model_failure_keeps_the_completed_file(tmp_path, monkeypatch):
    calls = []
    _patch_download(monkeypatch, tmp_path, calls, fail_on=2)
    names = list(models_mod.MODELS)

    try:
        models_mod.ensure_models()
    except urllib.error.URLError:
        pass
    else:
        raise AssertionError("ensure_models() swallowed the download error")

    assert (tmp_path / names[0]).is_file()
    assert not (tmp_path / names[1]).exists()
    assert list(tmp_path.glob("*.part")) == []
