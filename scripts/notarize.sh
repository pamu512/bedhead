#!/bin/bash
# Notarize the bedhead binary: sign -> notarytool -> staple -> verify.
# Prereqs (one-time, owner):
#   1. Developer ID Application cert in the keychain
#   2. xcrun notarytool store-credentials bedhead-notary --apple-id <id> --team-id 2NT7V479F6
# Usage: scripts/notarize.sh [path-to-binary]
set -euo pipefail

BIN="${1:-dist/bedhead}"
CERT=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed -E 's/.*\) ([A-F0-9]{40}).*/\1/' || true)
if [[ -z "$CERT" ]]; then
  echo "ERROR: no 'Developer ID Application' certificate in keychain." >&2
  echo "Create one at developer.apple.com -> Certificates, then rerun." >&2
  exit 1
fi

echo "==> signing with $CERT"
codesign --force --deep --options runtime --timestamp -s "$CERT" "$BIN"

echo "==> notarizing (can take minutes)"
DITTO=$(mktemp -d)/bedhead.zip
ditto -c -k --keepParent "$BIN" "$DITTO"
xcrun notarytool submit "$DITTO" --keychain-profile bedhead-notary --wait

echo "==> stapling"
xcrun stapler staple "$BIN"

echo "==> verifying"
spctl -a -t execute "$BIN"
codesign --verify --strict --verbose=2 "$BIN"
echo "OK: notarized binary at $BIN"
