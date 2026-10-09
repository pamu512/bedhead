# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the bedhead CLI (macOS, Linux, Windows).
# The Tk panel is optional (users can run python -m bedhead.panel from source),
# so it is excluded to keep the bundle lean; everything else is one-file.

import sysconfig

block_cipher = None

# site-packages of the BUILD venv, portable across platforms/CI
SITE = sysconfig.get_paths()["purelib"]

a = Analysis(
    ["bedhead/__main__.py"],
    pathex=["."],
    binaries=[
        # mediapipe.tasks.c is loaded dynamically by mediapipe's python code;
        # PyInstaller's static analysis misses the native libs and package data.
        (SITE + "/mediapipe/tasks/c/*", "mediapipe/tasks/c"),
    ],
    datas=[],
    hiddenimports=[
        "bedhead",
        "bedhead.cli",
        "bedhead.config",
        "bedhead.guard",
        "bedhead.autotune",
        "bedhead.colormatch",
        "bedhead.lighting",
        "bedhead.quality",
        "bedhead.retoucher",
        "bedhead.segmenter",
        "bedhead.sinks",
        "bedhead.tierb",
        "bedhead.tracker",
        "bedhead.clothes",
        "bedhead.benchmark",
        "bedhead.freqblend",
        "pyvirtualcam",
        "insightface.model_zoo",
    ],
    collect_all=["mediapipe", "onnxruntime"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "pandas"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="bedhead",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
