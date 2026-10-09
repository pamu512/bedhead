# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the bedhead CLI (macOS first; same spec works on Linux).
# The Tk panel is optional (users can run python -m bedhead.panel from source),
# so it is excluded to keep the bundle lean; everything else is one-file.

block_cipher = None

a = Analysis(
    ["bedhead/__main__.py"],
    pathex=["."],
    binaries=[
        # mediapipe.tasks.c is loaded dynamically by mediapipe's python code;
        # PyInstaller's static analysis misses the dylib and the package data.
        (
            ".venv/lib/python3.12/site-packages/mediapipe/tasks/c/*",
            "mediapipe/tasks/c",
        ),
    ],
    datas=[
        # ximgproc/guidedFilter needs the contrib opencv dylib; PyInstaller
        # finds cv2 automatically, but the mediapipe task-file stubs do not
        # ship (models download to the user cache at first run instead).
    ],
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
