"""aelock: exposure control availability + graceful no-op."""

from bedhead.aelock import _HAVE_PYOBJC, _default_device, lock_exposure, unlock_exposure


def test_module_imports_cleanly():
    # on any platform the module must import and report availability
    assert isinstance(_HAVE_PYOBJC, bool)


def test_default_device_is_cached_or_none():
    d1 = _default_device()
    d2 = _default_device()
    assert d1 is d2  # lru_cache


def test_lock_returns_bool_never_raises():
    r = lock_exposure()
    assert isinstance(r, bool)
    r2 = unlock_exposure()
    assert isinstance(r2, bool)
