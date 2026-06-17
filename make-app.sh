#!/bin/bash
# Build "Dictation App.app" around dictation.py so macOS recognises it by name
# in the Privacy lists and shows permission prompts with our usage strings.
#
# The trick: the bundle ships a *copy* of the Python interpreter as its own
# executable. macOS attributes TCC permissions to the .app that contains the
# running executable — so the process must live inside the bundle, not be an
# exec to /usr/bin/python3 (which would resolve to Apple's "Python" identity).
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"   # repo dir — where dictation.py lives
# Build OUTSIDE the repo: ~/Documents is iCloud/file-provider synced, which
# stamps xattrs (com.apple.fileprovider.*) that codesign refuses. ~/Applications
# is not synced. The bundle references the script in the repo by absolute path.
BUNDLE="$HOME/Applications/DictationApp.app"
DISPLAY_NAME="Dictation App"
BUNDLE_ID="com.jonatan.dictationapp"

# Resolve the active framework Python: its real binary (the framework's
# Python.app interpreter, not the /usr/bin stub), prefix, and the
# self-referential framework dylib load command we need to make absolute.
PYHOME="$(/usr/bin/python3 -c 'import sys; print(sys.prefix)')"
PYBIN="${PYHOME}/Resources/Python.app/Contents/MacOS/Python"
[ -x "$PYBIN" ] || PYBIN="$(/usr/bin/python3 -c 'import sys; print(sys.executable)')"
DYLIB="${PYHOME}/Python3"
OLD_LC="$(otool -L "$PYBIN" | awk '/@executable_path.*Python3/{print $1; exit}')"

echo "Building ${BUNDLE}"
echo "  interpreter: ${PYBIN}"
echo "  PYTHONHOME : ${PYHOME}"

mkdir -p "$HOME/Applications"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS"

# 1. Copy the interpreter in as the bundle's own executable and repoint its
#    framework load command at an absolute path (it's broken once relocated).
cp "$PYBIN" "$BUNDLE/Contents/MacOS/python3"
chmod u+w "$BUNDLE/Contents/MacOS/python3"
if [ -n "$OLD_LC" ]; then
    install_name_tool -change "$OLD_LC" "$DYLIB" "$BUNDLE/Contents/MacOS/python3"
fi

# 2. Launcher (the CFBundleExecutable): set PYTHONHOME, then exec the in-bundle
#    interpreter so the live process stays inside the bundle for both launchd
#    and double-click launches.
cat > "$BUNDLE/Contents/MacOS/DictationApp" <<LAUNCH
#!/bin/bash
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
export PYTHONHOME="${PYHOME}"
exec "\$DIR/python3" "${APP_DIR}/dictation.py"
LAUNCH
chmod +x "$BUNDLE/Contents/MacOS/DictationApp"

# 3. Info.plist — identity, menu-bar-only, and usage strings so macOS prompts.
cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>            <string>${DISPLAY_NAME}</string>
    <key>CFBundleDisplayName</key>     <string>${DISPLAY_NAME}</string>
    <key>CFBundleIdentifier</key>      <string>${BUNDLE_ID}</string>
    <key>CFBundleExecutable</key>      <string>DictationApp</string>
    <key>CFBundlePackageType</key>     <string>APPL</string>
    <key>CFBundleVersion</key>         <string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSUIElement</key>             <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Dictation App records audio so it can transcribe your speech.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>Dictation App pastes transcribed text into the active app.</string>
</dict>
</plist>
PLIST

# 4. Ad-hoc sign so permission grants persist across restarts, then register
#    with LaunchServices. Strip xattrs first (codesign refuses "detritus"),
#    and sign inside-out: the relocated interpreter, then the bundle.
xattr -cr "$BUNDLE"
codesign --force --sign - "$BUNDLE/Contents/MacOS/python3"
codesign --force --sign - "$BUNDLE"
plutil -lint "$BUNDLE/Contents/Info.plist" >/dev/null
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$BUNDLE" 2>/dev/null || true

echo "Built ${BUNDLE}  (identity: ${BUNDLE_ID})"
