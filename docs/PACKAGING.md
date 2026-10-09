# Building the standalone binary

The `bedhead` CLI ships as a single-file executable (PyInstaller) so a clean
machine needs no Python, no venv, no pip.

## Installers

- **macOS**: `scripts/build_mac_installer.sh` wraps the binary in a
  standards-compliant `.pkg` (pkgbuild, identifier `com.pamu512.bedhead`,
  installs to `/usr/local/bin`). `--sign`/`--notarize` flags sign with a
  Developer ID Installer cert and staple (same keychain profile as
  `scripts/notarize.sh`: `bedhead-notary`). CI builds and smokes it every
  push (artifact `bedhead-macos-pkg`).
- **Windows**: `installer/bedhead.iss` (Inno Setup) builds
  `bedhead-<version>-windows-x64.exe`: per-user install, adds the install
  dir to the user `PATH`, Start-menu shortcuts, uninstaller. CI builds it
  on `windows-latest` via chocolatey Inno Setup (artifact
  `bedhead-windows-installer`). Build locally: `ISCC.exe installer\bedhead.iss`.
  Note: pyvirtualcam on Windows uses the OBS virtual camera driver that
  ships with an OBS Studio install; the preview works without it.

## Build (macOS, from the repo root)

```bash
uv venv && uv pip install -e ".[guard,test]" pyinstaller
.venv/bin/pyinstaller bedhead.spec --noconfirm
./dist/bedhead --version
bash scripts/build_mac_installer.sh   # -> dist/bedhead-<version>-macos.pkg
```

Output: `dist/bedhead` (~160 MB; mediapipe + opencv-contrib + onnxruntime
are large but self-contained).

## What the binary does on first run

- downloads the MediaPipe models into `~/Library/Caches/bedhead/models`
  (Linux: `~/.cache/bedhead`), sha256-verified
- camera + screen-recording permissions still apply (macOS TCC); the
  Terminal/launcher must be allowed for Camera in Privacy & Security

## Packaging notes (hard-won)

- entry must be a module (`bedhead/__main__.py`) — a bare script breaks
  relative imports in the frozen app
- `mediapipe` needs `collect_all` **plus** an explicit binaries glob for
  `mediapipe/tasks/c/` (its C bindings + `libmediapipe.dylib` are loaded
  dynamically and invisible to static analysis)
- `matplotlib` must NOT be excluded (mediapipe imports it lazily)
- the Tk control panel is intentionally excluded; it stays a source-install
  extra (`python -m bedhead.panel`)

## Verify like CI does

```bash
env -i HOME=$(mktemp -d) PATH=/usr/bin:/bin ./dist/bedhead --version
```

A passing run prints `bedhead <version>` with no Python present.
