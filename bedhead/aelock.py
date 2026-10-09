"""Camera exposure control on macOS via native AVFoundation (pyobjc).

OpenCV's AVFoundation backend exposes NO exposure properties (all read 0),
so the pipeline cannot stop the camera's auto-exposure from fighting the
background-darken controller (measured: darken -> AE brightens the scene
-> closed loop chases its own tail). macOS camera settings are
device-global, so locking the device through a native AVCaptureDevice
reference affects OpenCV's capture too.

Usage:
    from bedhead.aelock import lock_exposure, unlock_exposure
    locked = lock_exposure()          # freeze AE at its current level
    ...
    unlock_exposure()                 # back to continuous AE

Hardware reality (measured on a MacBook Pro Camera, macOS 26):
the built-in DAL camera supports NO exposure or white-balance modes at all
(isExposureModeSupported_ returns False for locked/auto/custom; same for
white balance). External UVC cameras (DSLRs, capture cards) generally DO
support lock. This module is a no-op returning False on cameras without
support, and the closed-loop background_darken controller in
bedhead.autotune remains the fallback that works everywhere.
"""
from __future__ import annotations

import functools

import numpy as np

try:
    import AVFoundation

    _HAVE_PYOBJC = True
except Exception:  # pragma: no cover - non-macOS / pyobjc missing  # noqa: BLE001
    _HAVE_PYOBJC = False


@functools.lru_cache(maxsize=1)
def _default_device() -> AVFoundation.AVCaptureDevice | None:
    if not _HAVE_PYOBJC:
        return None
    try:
        devs = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(
            AVFoundation.AVMediaTypeVideo))
    except Exception:  # noqa: BLE001
        return None
    if not devs:
        return None
    # prefer a device whose name suggests the built-in camera
    for d in devs:
        if "camera" in str(d.localizedName()).lower():
            return d
    return devs[0]


def lock_exposure() -> bool:
    """Freeze the default camera's auto-exposure at its current level.

    Returns True if the lock was applied. Safe no-op (False) off-macOS or
    without pyobjc; the closed-loop controller remains the fallback.
    """
    dev = _default_device()
    if dev is None:
        return False
    try:
        ok, err = dev.lockForConfiguration()
        if not ok or err is not None:
            return False
        try:
            if dev.isExposureModeSupported_(AVFoundation.AVCaptureExposureModeLocked):
                dev.setExposureMode_(AVFoundation.AVCaptureExposureModeLocked)
                return True
            return False
        finally:
            dev.unlockForConfiguration()
    except Exception:  # noqa: BLE001
        return False


def set_exposure_bias(target: float) -> bool:
    """Set a custom exposure bias target (EV units, typically -2..2)."""
    dev = _default_device()
    if dev is None:
        return False
    try:
        ok, err = dev.lockForConfiguration()
        if not ok or err is not None:
            return False
        try:
            if dev.isExposureModeSupported_(AVFoundation.AVCaptureExposureModeCustom):
                dev.setExposureModeCustomWithDuration_ISO_(dev.activeFormat().minExposureDuration(),
                                                           dev.activeFormat().iso())
                dev.setExposureTargetBias_(np.float32(target))
                return True
            return False
        finally:
            dev.unlockForConfiguration()
    except Exception:  # noqa: BLE001
        return False


def unlock_exposure() -> bool:
    """Return the camera to continuous auto-exposure."""
    dev = _default_device()
    if dev is None:
        return False
    try:
        ok, err = dev.lockForConfiguration()
        if not ok or err is not None:
            return False
        try:
            dev.setExposureMode_(AVFoundation.AVCaptureExposureModeContinuousAutoExposure)
            return True
        finally:
            dev.unlockForConfiguration()
    except Exception:  # noqa: BLE001
        return False
