#!/bin/bash
# Build the bedhead macOS installer (.pkg) around the PyInstaller binary.
# Prereqs: ./dist/bedhead exists (run pyinstaller first), plus pkgbuild.
# Usage: scripts/build_mac_installer.sh [--sign <CERT> | --notarize]
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])" 2>/dev/null || echo "0.1.0")
IDENTIFIER="com.pamu512.bedhead"
INSTALL_ROOT="$(mktemp -d)/root"
DEST="/usr/local/bin"

if [[ ! -x dist/bedhead ]]; then
  echo "ERROR: dist/bedhead not found. Run: pyinstaller bedhead.spec --noconfirm" >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT$DEST"
cp dist/bedhead "$INSTALL_ROOT$DEST/bedhead"

# scripts run at INSTALL time: warn about camera permission upfront
POSTINSTALL="$(mktemp -d)/postinstall"
cat > "$POSTINSTALL" <<'EOS'
#!/bin/bash
# bedhead postinstall: no privileged actions needed; just friendly output.
echo "bedhead installed to /usr/local/bin/bedhead"
echo "First run will ask for Camera permission (macOS Privacy & Security)."
echo "Models (~20 MiB) download once into ~/Library/Caches/bedhead."
EOS
chmod +x "$POSTINSTALL"

echo "==> building pkg (version $VERSION)"
PKG="dist/bedhead-$VERSION-macos.pkg"
mkdir -p dist
pkgbuild \
  --root "$INSTALL_ROOT" \
  --identifier "$IDENTIFIER" \
  --version "$VERSION" \
  --install-location "/" \
  --scripts "$(dirname "$POSTINSTALL")" \
  "$PKG"

if [[ "${1:-}" == "--sign" || "${1:-}" == "--notarize" ]]; then
  CERT="${2:-}"
  if [[ -z "$CERT" ]]; then
    CERT=$(security find-identity -v -p codesigning | grep "Developer ID Installer" | head -1 | sed -E 's/.*\) ([A-F0-9]{40}).*/\1/' || true)
  fi
  if [[ -z "$CERT" ]]; then
    echo "ERROR: no 'Developer ID Installer' certificate found." >&2; exit 1
  fi
  echo "==> signing pkg with $CERT"
  productsign --sign "$CERT" "$PKG" "$PKG.signed"
  mv "$PKG.signed" "$PKG"
fi

if [[ "${1:-}" == "--notarize" ]]; then
  echo "==> notarizing"
  xcrun notarytool submit "$PKG" --keychain-profile bedhead-notary --wait
  xcrun stapler staple "$PKG"
fi

echo "==> done: $PKG"
