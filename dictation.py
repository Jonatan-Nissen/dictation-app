#!/usr/bin/env python3
"""
System-wide macOS dictation tool.

Press Cmd+Shift+D to start recording, press again to stop.
Transcribes via OpenAI Whisper API and pastes into the active text input.
"""

import os
import sys
import subprocess
import tempfile
import threading
import time

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from pynput import keyboard

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
    print("ERROR: Set your OPENAI_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

SAMPLE_RATE = 16000  # 16 kHz mono — what Whisper expects
MODEL = "gpt-4o-mini-transcribe"
HOTKEY = "<cmd>+<shift>+d"

client = OpenAI(api_key=OPENAI_API_KEY)

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
    script = (
        f'display notification "{message}" with title "{title}"'
    )
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
        print(f"sounddevice status: {status}", file=sys.stderr)
    audio_chunks.append(indata.copy())


def start_recording() -> None:
    global stream, audio_chunks
    audio_chunks = []
    play_sound("Bottle")
    notify("Dictation", "🎙 Recording… press Cmd+Shift+D to stop")
    print("🎙  Recording… press Cmd+Shift+D to stop")
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
    global stream
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
    play_sound("Bottle")
    notify("Dictation", "⏹ Stopped — transcribing…")
    print("⏹  Stopped recording — transcribing…")

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray) -> str:
    """Write audio to a temp WAV file, send to OpenAI, return text."""
    tmp = os.path.join(tempfile.gettempdir(), "dictation_recording.wav")
    wavfile.write(tmp, SAMPLE_RATE, audio)

    with open(tmp, "rb") as f:
        result = client.audio.transcriptions.create(
            model=MODEL,
            file=f,
            response_format="text",
        )

    os.remove(tmp)
    return result.strip() if isinstance(result, str) else result.text.strip()

# ---------------------------------------------------------------------------
# Text injection (clipboard + Cmd+V, then restore)
# ---------------------------------------------------------------------------

def _get_clipboard() -> str:
    """Read current clipboard contents (plain text)."""
    try:
        r = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2)
        return r.stdout
    except Exception:
        return ""


def _set_clipboard(text: str) -> None:
    """Write text to the clipboard."""
    try:
        subprocess.run(
            ["pbcopy"], input=text, text=True, timeout=2, check=True,
        )
    except Exception as e:
        print(f"pbcopy error: {e}", file=sys.stderr)


def inject_text(text: str) -> None:
    """Paste text into the currently focused input, then restore clipboard."""
    original_clipboard = _get_clipboard()

    _set_clipboard(text)
    time.sleep(0.05)  # let pasteboard propagate

    # Simulate Cmd+V
    kb = keyboard.Controller()
    with kb.pressed(keyboard.Key.cmd):
        kb.press("v")
        kb.release("v")

    time.sleep(0.25)  # wait for paste to complete

    # Restore original clipboard
    _set_clipboard(original_clipboard)

# ---------------------------------------------------------------------------
# Toggle handler (runs transcription in a thread)
# ---------------------------------------------------------------------------

def _process_recording() -> None:
    """Transcribe the recorded audio and inject the result."""
    if not audio_chunks:
        print("No audio captured.", file=sys.stderr)
        return

    audio = np.concatenate(audio_chunks, axis=0)
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.3:
        print("Recording too short, skipping.", file=sys.stderr)
        return

    try:
        text = transcribe(audio)
        if text:
            print(f"✅ Transcribed: {text}")
            # Show a short preview in the notification
            preview = text[:80] + ("…" if len(text) > 80 else "")
            notify("Dictation", f"✅ {preview}")
            inject_text(text)
        else:
            print("⚠️  Empty transcription result.", file=sys.stderr)
    except Exception as e:
        print(f"❌ Transcription error: {e}", file=sys.stderr)


def on_toggle() -> None:
    global is_recording
    with lock:
        if not is_recording:
            is_recording = True
            start_recording()
        else:
            is_recording = False
            stop_recording()
            # Process in background thread so hotkey listener stays responsive
            threading.Thread(target=_process_recording, daemon=True).start()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 50)
    print("  Dictation Tool")
    print(f"  Hotkey: {HOTKEY}")
    print(f"  Model:  {MODEL}")
    print("  Press Ctrl+C to quit")
    print("=" * 50)

    with keyboard.GlobalHotKeys({HOTKEY: on_toggle}) as hotkey_listener:
        try:
            hotkey_listener.join()
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
