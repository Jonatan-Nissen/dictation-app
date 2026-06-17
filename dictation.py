#!/usr/bin/env python3
"""
System-wide macOS dictation tool.

Press Cmd+Shift+D to start recording, press again to stop.
Transcribes via OpenAI Whisper API and pastes into the active text input.

Designed to run always-on in the background:
  - Single-instance lock prevents duplicate processes fighting over the
    hotkey and microphone (the #1 cause of "hotkey stopped working").
  - Every hotkey callback is crash-guarded so a transient device error
    never kills the listener thread.
  - Text is pasted via osascript (not pynput's Controller) to avoid
    desyncing the global-hotkey listener's modifier state.
  - All activity is logged to dictation.log for after-the-fact diagnosis.
  - A menu-bar icon shows the tool is alive (mic glyph), turning red while
    recording. The NSApplication runs on the main thread; the hotkey
    listener runs in a background thread.
"""

import fcntl
import logging
import os
import signal
import sys
import subprocess
import tempfile
import threading
import time
from logging.handlers import RotatingFileHandler

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from pynput import keyboard

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSColor,
    NSImage,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSVariableStatusItemLength,
    NSTimer,
)
from Foundation import NSObject

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SAMPLE_RATE = 16000  # 16 kHz mono — what Whisper expects
MODEL = "gpt-4o-mini-transcribe"
HOTKEY = "<cmd>+<shift>+d"
HOTKEY_DISPLAY = "⌘⇧D"

# Menu-bar glyphs (SF Symbols, with emoji fallback if unavailable)
IDLE_SYMBOL = "mic"
REC_SYMBOL = "mic.fill"
IDLE_EMOJI = "🎙"
REC_EMOJI = "🔴"

LOCK_PATH = os.path.join(tempfile.gettempdir(), "dictation.lock")
LOG_PATH = os.path.join(APP_DIR, "dictation.log")

# ---------------------------------------------------------------------------
# Logging — file (always-on diagnosis) + stderr (foreground use)
# ---------------------------------------------------------------------------

log = logging.getLogger("dictation")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s")

_file_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
_file_handler.setFormatter(_fmt)
log.addHandler(_file_handler)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(_fmt)
log.addHandler(_stderr_handler)

# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------

_lock_handle = None  # kept open for the process lifetime


def acquire_single_instance_lock() -> bool:
    """Take an exclusive lock so only one dictation process runs at a time."""
    global _lock_handle
    _lock_handle = open(LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    _lock_handle.truncate(0)
    _lock_handle.write(str(os.getpid()))
    _lock_handle.flush()
    return True


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

is_recording = False
audio_chunks = []
stream = None
lock = threading.Lock()

# ---------------------------------------------------------------------------
# Audio + visual feedback
# ---------------------------------------------------------------------------

def play_sound(name: str) -> None:
    """Play a macOS system sound using NSSound (works reliably from background scripts)."""
    try:
        from AppKit import NSSound
        sound = NSSound.alloc().initWithContentsOfFile_byReference_(
            f"/System/Library/Sounds/{name}.aiff", True
        )
        if sound:
            sound.play()
    except ImportError:
        # Fallback to afplay if AppKit not available
        def _play():
            path = f"/System/Library/Sounds/{name}.aiff"
            if os.path.exists(path):
                subprocess.run(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Thread(target=_play, daemon=True).start()


def notify(title: str, message: str) -> None:
    """Show a macOS notification banner."""
    # Escape double quotes so the AppleScript string stays well-formed.
    safe = message.replace('"', '\\"')
    script = f'display notification "{safe}" with title "{title}"'
    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    if status:
        log.warning("sounddevice status: %s", status)
    audio_chunks.append(indata.copy())


def _cleanup_stream() -> None:
    """Stop and close the input stream, ignoring errors. Safe to call anytime."""
    global stream
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            log.warning("error closing stream: %s", e)
        finally:
            stream = None


def start_recording() -> None:
    """Open the mic and begin capturing. Raises if the device can't be opened."""
    global stream, audio_chunks
    audio_chunks = []
    play_sound("Bottle")
    notify("Dictation", "🎙 Recording… press Cmd+Shift+D to stop")
    log.info("recording started")
    # Delay so the sound finishes before the audio input stream takes over
    time.sleep(0.8)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=_audio_callback,
    )
    stream.start()


def stop_recording() -> None:
    _cleanup_stream()
    play_sound("Bottle")
    notify("Dictation", "⏹ Stopped — transcribing…")
    log.info("recording stopped — transcribing")

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray) -> str:
    """Write audio to a temp WAV file, send to OpenAI, return text."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    tmp = os.path.join(tempfile.gettempdir(), "dictation_recording.wav")
    wavfile.write(tmp, SAMPLE_RATE, audio)

    try:
        with open(tmp, "rb") as f:
            result = client.audio.transcriptions.create(
                model=MODEL,
                file=f,
                response_format="text",
            )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return result.strip() if isinstance(result, str) else result.text.strip()

# ---------------------------------------------------------------------------
# Text injection (clipboard + Cmd+V via osascript)
# ---------------------------------------------------------------------------

def _set_clipboard(text: str) -> None:
    """Write text to the clipboard."""
    try:
        subprocess.run(
            ["pbcopy"], input=text, text=True, timeout=2, check=True,
        )
    except Exception as e:
        log.warning("pbcopy error: %s", e)


def inject_text(text: str) -> None:
    """Put the transcription on the clipboard and paste it into the focused input.

    The text is intentionally LEFT on the clipboard afterwards (the original is
    NOT restored) so every dictation lands in clipboard history — e.g. JumpCut —
    and can be re-pasted if the paste missed the intended field. Running as an
    app there's no terminal scrollback to recover it from otherwise.

    Uses osascript to send Cmd+V rather than pynput's Controller: injecting
    synthetic key events through pynput while its GlobalHotKeys listener is
    running can desync the listener and silently break the hotkey.
    """
    _set_clipboard(text)
    time.sleep(0.05)  # let pasteboard propagate

    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "v" using command down'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(0.25)  # wait for paste to complete
    # Leave `text` on the clipboard so it stays in clipboard history.

# ---------------------------------------------------------------------------
# Toggle handler (runs transcription in a thread)
# ---------------------------------------------------------------------------

def _process_recording() -> None:
    """Transcribe the recorded audio and inject the result."""
    if not audio_chunks:
        log.warning("no audio captured")
        return

    audio = np.concatenate(audio_chunks, axis=0)
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.3:
        log.info("recording too short (%.2fs), skipping", duration)
        return

    try:
        text = transcribe(audio)
        if text:
            log.info("transcribed: %s", text)
            # Show a short preview in the notification
            preview = text[:80] + ("…" if len(text) > 80 else "")
            notify("Dictation", f"✅ {preview}")
            inject_text(text)
        else:
            log.warning("empty transcription result")
            notify("Dictation", "⚠️ Empty transcription")
    except Exception as e:
        log.exception("transcription error")
        notify("Dictation", f"❌ Transcription error: {e}")


def _handle_toggle() -> None:
    """Start/stop recording. Runs on a worker thread (see on_toggle)."""
    global is_recording
    with lock:
        try:
            if not is_recording:
                start_recording()
                is_recording = True  # only after the stream actually opened
            else:
                is_recording = False
                stop_recording()
                # Process in background thread so hotkey listener stays responsive
                threading.Thread(target=_process_recording, daemon=True).start()
        except Exception as e:
            log.exception("toggle failed — resetting state")
            is_recording = False
            _cleanup_stream()
            notify("Dictation", f"⚠️ Error: {e}")


def on_toggle() -> None:
    """Hotkey callback — runs inside the macOS event-tap, so it MUST return fast.

    Starting a recording sleeps ~0.8s and opens an audio device; doing that
    here would block the event tap long enough for macOS to disable it
    (kCGEventTapDisabledByTimeout), silently killing the hotkey until restart.
    Hand the slow work to a worker thread and return immediately.
    """
    threading.Thread(target=_handle_toggle, daemon=True).start()

# ---------------------------------------------------------------------------
# Hotkey listener (runs in a background thread under the Cocoa app)
# ---------------------------------------------------------------------------

def run_hotkey_listener() -> None:
    """Listen for the global hotkey, restarting the listener if it ever dies."""
    while True:
        try:
            with keyboard.GlobalHotKeys({HOTKEY: on_toggle}) as hotkey_listener:
                hotkey_listener.join()
        except Exception:
            # An always-on process alive but deaf to the hotkey is the worst
            # failure mode — restart the listener rather than give up.
            log.exception("hotkey listener crashed — restarting in 2s")
            time.sleep(2)

# ---------------------------------------------------------------------------
# Menu-bar status item
# ---------------------------------------------------------------------------

def _symbol_image(name: str):
    """Build a template NSImage from an SF Symbol, or None if unavailable."""
    try:
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "Dictation")
        if img is not None:
            img.setTemplate_(True)
        return img
    except Exception:
        return None


class DictationStatusDelegate(NSObject):
    """Owns the menu-bar item and mirrors recording state into its icon."""

    def applicationDidFinishLaunching_(self, _notification) -> None:
        try:
            self._idle_img = _symbol_image(IDLE_SYMBOL)
            self._rec_img = _symbol_image(REC_SYMBOL)
            self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(
                NSVariableStatusItemLength
            )
            self._shown = None
            self.render_(False)

            menu = NSMenu.alloc().init()
            for label in (f"Dictation — {MODEL}", f"Hotkey: {HOTKEY_DISPLAY}"):
                info = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label, None, "")
                info.setEnabled_(False)
                menu.addItem_(info)
            menu.addItem_(NSMenuItem.separatorItem())
            menu.addItem_(
                NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    "Quit Dictation", "terminate:", "q"
                )
            )
            self.item.setMenu_(menu)

            # Poll shared state on the main thread (AppKit is not thread-safe,
            # so the hotkey thread must not touch the status item directly).
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.3, self, b"tick:", None, True
            )
            log.info("menu-bar item ready")
        except Exception:
            log.exception("failed to build menu-bar item (hotkey still active)")

    def tick_(self, _timer) -> None:
        if is_recording != self._shown:
            self.render_(is_recording)

    def render_(self, recording: bool) -> None:
        self._shown = recording
        button = self.item.button()
        img = self._rec_img if recording else self._idle_img
        if img is not None:
            button.setImage_(img)
            button.setTitle_("")
            button.setContentTintColor_(NSColor.systemRedColor() if recording else None)
        else:  # SF Symbols unavailable — fall back to emoji title
            button.setImage_(None)
            button.setTitle_(REC_EMOJI if recording else IDLE_EMOJI)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        log.error("Set your OPENAI_API_KEY in .env")
        sys.exit(1)

    if not acquire_single_instance_lock():
        # A duplicate launch is benign (e.g. launchd plus a manual start).
        # Exit 0 so launchd's KeepAlive does not treat it as a crash and loop.
        log.warning(
            "Another dictation instance already holds the lock (%s). Exiting.",
            LOCK_PATH,
        )
        sys.exit(0)

    log.info("=" * 50)
    log.info("Dictation Tool  |  hotkey %s  |  model %s  |  pid %d",
             HOTKEY, MODEL, os.getpid())
    log.info("Logging to %s", LOG_PATH)
    log.info("=" * 50)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # menu bar, no Dock icon
    delegate = DictationStatusDelegate.alloc().init()
    app.setDelegate_(delegate)

    # Quit cleanly on SIGTERM/SIGINT (launchd stop, Ctrl+C). The 0.3s timer
    # gives the interpreter a chance to deliver these into the Cocoa run loop.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.terminate_(None))

    threading.Thread(target=run_hotkey_listener, daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()
