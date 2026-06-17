#!/bin/bash
# Install (or reinstall) the dictation tool as a per-user LaunchAgent:
# starts at login, restarts on crash, runs as a single instance.
set -euo pipefail

LABEL="com.jonatan.dictation"
SRC="$(cd "$(dirname "$0")" && pwd)/${LABEL}.plist"
DEST="$HOME/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

echo "Installing ${LABEL} LaunchAgent…"

# Stop any running instance (managed or manual) for a clean cutover.
launchctl bootout "${DOMAIN}/${LABEL}" 2>/dev/null || true
pkill -f "dictation-app/dictation.py" 2>/dev/null || true
sleep 1

mkdir -p "$HOME/Library/LaunchAgents"
cp "$SRC" "$DEST"

launchctl bootstrap "$DOMAIN" "$DEST"
launchctl enable "${DOMAIN}/${LABEL}"
launchctl kickstart -k "${DOMAIN}/${LABEL}"

echo "Done. Status:"
launchctl print "${DOMAIN}/${LABEL}" | grep -E "state|pid|program =" || true

cat <<'NOTE'

If the menu-bar mic icon appears but the Cmd+Shift+D hotkey does nothing,
grant /usr/bin/python3 permission in System Settings → Privacy & Security:
  • Accessibility        (for the global hotkey + paste)
  • Input Monitoring     (for the global hotkey)
Then run:  launchctl kickstart -k gui/$(id -u)/com.jonatan.dictation

To stop/uninstall:
  launchctl bootout gui/$(id -u)/com.jonatan.dictation
  rm ~/Library/LaunchAgents/com.jonatan.dictation.plist
NOTE
