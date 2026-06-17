# dictation-app

Script that lets me dictate and transcribe into any text input on macOS.

Press the hotkey to start recording, press again to stop. Audio is sent to
OpenAI for transcription and pasted into the focused field.

- **Hotkey:** `⌘⇧D` (`<cmd>+<shift>+d`)
- **Model:** `gpt-4o-mini-transcribe`
- **Menu bar:** a mic icon shows the tool is running; it turns red while recording.

## Run manually

```sh
python3 ~/Documents/GitHub/base/repos/dictation-app/dictation.py
```

Quit from the menu-bar icon (**Quit Dictation**) or with `Ctrl+C`.

## Run always-on (recommended)

Install as a per-user LaunchAgent — starts at login, restarts on crash,
single instance:

```sh
./install-launchagent.sh
```

The first time, macOS may require granting `/usr/bin/python3` permission under
**System Settings → Privacy & Security**:

- **Accessibility** — for the global hotkey and paste
- **Input Monitoring** — for the global hotkey

Then kickstart it: `launchctl kickstart -k gui/$(id -u)/com.jonatan.dictation`

Uninstall:

```sh
launchctl bootout gui/$(id -u)/com.jonatan.dictation
rm ~/Library/LaunchAgents/com.jonatan.dictation.plist
```

## Robustness notes

- **Single-instance lock** (`fcntl.flock`) prevents duplicate processes from
  fighting over the hotkey and microphone.
- Hotkey callbacks are crash-guarded so a transient audio-device error can't
  kill the listener; the listener also auto-restarts if it ever dies.
- Paste uses `osascript` (not pynput's `Controller`) to avoid desyncing the
  global-hotkey listener.
- Activity is logged to `dictation.log` (rotating) for diagnosis.

## Setup

```sh
pip3 install -r requirements.txt   # AppKit/PyObjC ships with macOS system Python
cp .env.example .env               # then add your OPENAI_API_KEY
```
