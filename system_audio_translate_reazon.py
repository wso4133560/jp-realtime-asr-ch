#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
BYTES_PER_SAMPLE = 2


EVENT_TOKEN_RE = re.compile(r"^[\s\W_]+|[\s\W_]+$")
MUSIC_PREFIX_RE = re.compile(r"^[🎵🎶🎼♪♫♬]+")
DISALLOWED_TRANSLATION_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7afA-Za-z]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture live system audio, transcribe it with ReazonSpeech K2, and translate it to Chinese with Ollama."
    )
    parser.add_argument(
        "--source",
        default="auto",
        help="PulseAudio source name. Use 'auto' to detect the default source/monitor.",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available PulseAudio sources and exit.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("REAZON_DEVICE", "cuda"),
        help="ReazonSpeech device/provider, for example cuda or cpu.",
    )
    parser.add_argument(
        "--precision",
        default=os.getenv("REAZON_PRECISION", "fp32"),
        choices=["fp32", "int8", "int8-fp32"],
        help="ReazonSpeech model precision.",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("REAZON_LANGUAGE", "ja"),
        choices=["ja", "ja-en", "ja-en-mls-5k"],
        help="ReazonSpeech model language.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=8.0,
        help="Recognition window size in seconds.",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=1.5,
        help="How often to run recognition in seconds.",
    )
    parser.add_argument(
        "--min-window-sec",
        type=float,
        default=2.0,
        help="Minimum buffered audio before the first recognition pass.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.0015,
        help="Skip inference when RMS is below this threshold.",
    )
    parser.add_argument(
        "--silence-reset-sec",
        type=float,
        default=3.0,
        help="Reset incremental text state after this much silence.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Optional auto-stop duration for testing. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--print-mode",
        choices=["incremental", "full"],
        default="incremental",
        help="Print only novel suffixes or the full current source ASR result.",
    )
    parser.add_argument(
        "--print-source",
        action="store_true",
        help="Print source-language ASR lines in addition to Chinese translation lines.",
    )
    parser.add_argument(
        "--output-txt",
        default="",
        help="Optional path to append timestamped Chinese translation lines as plain text.",
    )
    parser.add_argument(
        "--output-source-txt",
        default="",
        help="Optional path to append timestamped source-language ASR lines as plain text.",
    )
    parser.add_argument(
        "--translate-model",
        default=os.getenv("TRANSLATE_MODEL", "translategemma:4b-it-q4_K_M"),
        help="Ollama model used for translation.",
    )
    parser.add_argument(
        "--translate-prompt",
        default="",
        help="Optional custom translation instruction. The recognized text will be appended after a blank line.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        help="Base URL of the local Ollama server.",
    )
    parser.add_argument(
        "--ollama-timeout-sec",
        type=float,
        default=20.0,
        help="Timeout for each Ollama translation request.",
    )
    parser.add_argument(
        "--no-output-warning-sec",
        type=float,
        default=5.0,
        help="Warn if no source or translation output is produced within this many seconds after capture starts.",
    )
    parser.add_argument(
        "--translate-min-chars",
        type=int,
        default=4,
        help="Skip translation for cleaned ASR text shorter than this many characters.",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Print capture RMS and raw ASR text diagnostics.",
    )
    return parser.parse_args()


def preload_cuda_runtime_libs(enabled):
    if not enabled.startswith("cuda"):
        return

    root_dir = Path(__file__).resolve().parent
    lib_paths = [
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops.so.9",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_adv.so.9",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_graph.so.9",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_heuristic.so.9",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
        root_dir / "venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9",
    ]
    missing = [str(path) for path in lib_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing CUDA runtime libraries in venv. Expected to find: " + ", ".join(missing)
        )

    for path in lib_paths:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


from reazonspeech.k2.asr import audio_from_numpy, load_model, transcribe  # noqa: E402


def normalize_device(device):
    if device.startswith("cuda"):
        return "cuda"
    return device


def run_pactl(*args):
    result = subprocess.run(
        ["pactl", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def list_sources():
    output = run_pactl("list", "short", "sources").strip()
    return output.splitlines() if output else []


def detect_default_source():
    for line in run_pactl("info").splitlines():
        if line.startswith("默认信源：") or line.startswith("Default Source:"):
            return line.split("：", 1)[-1].split(":", 1)[-1].strip()
    raise RuntimeError("Failed to detect the default PulseAudio source.")


def detect_capture_source(source_arg):
    if source_arg != "auto":
        return source_arg

    default_source = detect_default_source()
    if default_source.endswith(".monitor"):
        return default_source

    for line in run_pactl("list", "short", "sinks").strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == default_source:
            return f"{default_source}.monitor"
    return default_source


def normalize_text(text):
    return re.sub(r"\s+", "", text).strip()


def novel_suffix(previous, current):
    if not current or current == previous:
        return ""
    if not previous:
        return current
    if previous in current:
        return current.split(previous, 1)[1]

    max_overlap = min(len(previous), len(current))
    for overlap in range(max_overlap, 0, -1):
        if previous[-overlap:] == current[:overlap]:
            if overlap >= 4:
                return current[overlap:]
            break
    return current


def rms_level(samples):
    if samples.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def now_label():
    return time.strftime("%H:%M:%S")


def clean_text_for_translation(text):
    cleaned = text.strip()
    cleaned = MUSIC_PREFIX_RE.sub("", cleaned)
    cleaned = EVENT_TOKEN_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_translation_candidate(text, min_chars):
    cleaned = clean_text_for_translation(text)
    return cleaned, len(cleaned) >= min_chars


def clean_translation_output(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^[\s\"'「」『』【】()（）]+|[\s\"'「」『』【】()（）]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def has_disallowed_translation_script(text):
    return bool(DISALLOWED_TRANSLATION_RE.search(text))


@dataclass
class RecognitionState:
    last_text: str = ""
    best_text: str = ""
    silent_for_sec: float = 0.0
    last_emit_text: str = ""
    last_translation: str = ""
    last_translation_source: str = ""
    saw_output: bool = False
    warned_no_output: bool = False
    last_rms: float = 0.0


class SystemAudioChineseTranslator:
    def __init__(self, args):
        self.args = args
        self.capture_source = detect_capture_source(args.source)
        self.running = True
        self._prepare_output_files()
        print("status=loading_asr_model", flush=True)
        self.model = self._load_model()
        print("status=asr_model_ready", flush=True)

    def _prepare_output_files(self):
        for output_path in [self.args.output_txt, self.args.output_source_txt]:
            if output_path:
                parent = os.path.dirname(os.path.abspath(output_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(output_path, "a", encoding="utf-8"):
                    pass

    def _load_model(self):
        return load_model(
            device=normalize_device(self.args.device),
            precision=self.args.precision,
            language=self.args.language,
        )

    def _capture_command(self):
        return [
            "parec",
            "--device",
            self.capture_source,
            "--format=s16le",
            f"--rate={DEFAULT_RATE}",
            f"--channels={DEFAULT_CHANNELS}",
            "--raw",
        ]

    def _transcribe(self, samples):
        audio = audio_from_numpy(samples, DEFAULT_RATE)
        result = transcribe(self.model, audio)
        return normalize_text(result.text)

    def _debug_audio(self, rms, audio_sec, raw_text=""):
        if not self.args.debug_audio:
            return
        message = f"[{now_label()}] debug audio={audio_sec:0.1f}s rms={rms:0.4f}"
        if raw_text:
            message += f" raw={raw_text}"
        print(message, flush=True)

    def _append_translation_txt(self, translated_text):
        if not self.args.output_txt:
            return
        with open(self.args.output_txt, "a", encoding="utf-8") as handle:
            handle.write(f"[{now_label()}] {translated_text}\n")

    def _append_source_txt(self, source_text):
        if not self.args.output_source_txt:
            return
        with open(self.args.output_source_txt, "a", encoding="utf-8") as handle:
            handle.write(f"[{now_label()}] {source_text}\n")

    def _build_translation_prompt(self, text, strict=False):
        if self.args.translate_prompt:
            prompt = self.args.translate_prompt.strip()
            if strict:
                prompt += (
                    "\n\nYour previous answer contained non-Chinese script. "
                    "Reply again using only Simplified Chinese, Arabic numerals, and Chinese punctuation."
                )
            return f"{prompt}\n\n{text}"
        if strict:
            return (
                "You are a translation engine. "
                "Re-translate the source text into natural Simplified Chinese. "
                "Output exactly one line using only Simplified Chinese, Arabic numerals, and Chinese punctuation. "
                "Do not output Korean Hangul, Japanese Kana, romaji, English words, notes, labels, or the source text. "
                "Preserve names, numbers, dates, and proper nouns accurately. "
                f"<source>{text}</source>"
            )
        return (
            "You are a translation engine. "
            "Translate the source text into natural Simplified Chinese. "
            "Output exactly one line in Simplified Chinese only. "
            "Do not explain. Do not add notes. Do not repeat the source text. "
            "Do not output Korean Hangul, Japanese Kana, romaji, or English words. "
            "Preserve names, numbers, dates, currencies, and proper nouns accurately. "
            "If the source is already in Simplified Chinese, return it directly. "
            f"<source>{text}</source>"
        )

    def _ollama_generate(self, prompt):
        request_body = json.dumps(
            {
                "model": self.args.translate_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.args.ollama_url}/api/generate",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.ollama_timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return clean_translation_output(payload.get("response", ""))

    def _translate_text(self, text):
        translated = self._ollama_generate(self._build_translation_prompt(text, strict=False))
        if translated and has_disallowed_translation_script(translated):
            translated = self._ollama_generate(self._build_translation_prompt(text, strict=True))
        return translated

    def _is_forward_progress(self, state, text):
        if not text or text == state.last_emit_text:
            return False
        if not state.best_text:
            return True
        if len(text) < len(state.best_text):
            return False
        if state.best_text in text:
            return True

        max_overlap = min(len(state.best_text), len(text))
        for overlap in range(max_overlap, 0, -1):
            if state.best_text[-overlap:] == text[:overlap]:
                return overlap >= 4 and len(text) > overlap
        return False

    def _emit_output(self, state, text, audio_sec, infer_sec):
        if self.args.print_mode == "full":
            source_message = text
        else:
            source_message = novel_suffix(state.last_text, text)
            if not source_message and len(text) > len(state.last_text):
                source_message = text

        if self.args.print_source and source_message:
            print(
                f"[{now_label()}] source audio={audio_sec:0.1f}s infer={infer_sec:0.3f}s text={source_message}",
                flush=True,
            )
            state.saw_output = True

        forward_progress = self._is_forward_progress(state, text)
        if forward_progress:
            self._append_source_txt(text)
            state.last_emit_text = text
            state.best_text = text

        cleaned_text, should_translate = is_translation_candidate(
            text, self.args.translate_min_chars
        )
        if not should_translate:
            if self.args.debug_audio:
                print(
                    f"[{now_label()}] debug skip_translation raw={text} cleaned={cleaned_text}",
                    flush=True,
                )
            state.last_text = text
            return

        stable_translation = forward_progress or (
            source_message == text and len(cleaned_text) >= max(18, self.args.translate_min_chars)
        )
        if not stable_translation:
            if self.args.debug_audio:
                print(
                    f"[{now_label()}] debug skip_translation_not_stable raw={text}",
                    flush=True,
                )
            state.last_text = text
            return

        if cleaned_text == state.last_translation_source:
            state.last_text = text
            return

        try:
            translate_started = time.perf_counter()
            translated_text = self._translate_text(cleaned_text)
            translate_sec = time.perf_counter() - translate_started
        except Exception as exc:
            print(f"[{now_label()}] translation_error={exc}", flush=True)
            state.last_text = text
            return

        state.last_translation_source = cleaned_text
        if translated_text and translated_text != state.last_translation:
            print(
                f"[{now_label()}] translation model={self.args.translate_model} infer={translate_sec:0.3f}s text={translated_text}",
                flush=True,
            )
            self._append_translation_txt(translated_text)
            state.last_translation = translated_text
            state.saw_output = True

        state.last_text = text

    def run(self):
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        bytes_per_second = DEFAULT_RATE * DEFAULT_CHANNELS * BYTES_PER_SAMPLE
        window_bytes = int(self.args.window_sec * bytes_per_second)
        min_window_bytes = int(self.args.min_window_sec * bytes_per_second)
        step_bytes = max(1, int(self.args.step_sec * bytes_per_second))
        read_bytes = max(4096, bytes_per_second // 5)

        print(f"capture_source={self.capture_source}", flush=True)
        print(f"device={self.args.device}", flush=True)
        print(
            f"window_sec={self.args.window_sec} step_sec={self.args.step_sec} print_mode={self.args.print_mode}",
            flush=True,
        )
        print(f"translate_model={self.args.translate_model}", flush=True)
        if self.args.output_txt:
            print(f"output_txt={self.args.output_txt}", flush=True)
        if self.args.output_source_txt:
            print(f"output_source_txt={self.args.output_source_txt}", flush=True)
        if self.args.debug_audio:
            print("debug_audio=true", flush=True)

        process = subprocess.Popen(
            self._capture_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("ready=true", flush=True)
        state = RecognitionState()
        buffer = bytearray()
        pending = 0
        started_at = time.monotonic()

        try:
            while self.running:
                capture_elapsed_sec = time.monotonic() - started_at
                if (
                    not state.saw_output
                    and not state.warned_no_output
                    and capture_elapsed_sec >= self.args.no_output_warning_sec
                ):
                    print(
                        f"[{now_label()}] warning=no_output_yet source={self.capture_source} last_rms={state.last_rms:0.4f} hint=check_playback_after_ready_or_try_source_argument",
                        flush=True,
                    )
                    state.warned_no_output = True

                if self.args.duration_sec > 0 and capture_elapsed_sec >= self.args.duration_sec:
                    break

                chunk = process.stdout.read(read_bytes)
                if not chunk:
                    stderr = process.stderr.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"parec stopped unexpectedly: {stderr.strip()}")

                buffer.extend(chunk)
                pending += len(chunk)
                if len(buffer) > window_bytes:
                    del buffer[:-window_bytes]

                if len(buffer) < min_window_bytes or pending < step_bytes:
                    continue

                pending = 0
                samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                rms = rms_level(samples)
                state.last_rms = rms
                audio_sec = len(samples) / DEFAULT_RATE

                if rms < self.args.silence_threshold:
                    self._debug_audio(rms, audio_sec, raw_text="silence")
                    state.silent_for_sec += self.args.step_sec
                    if state.silent_for_sec >= self.args.silence_reset_sec:
                        state.last_text = ""
                        state.best_text = ""
                        state.last_emit_text = ""
                        state.last_translation = ""
                        state.last_translation_source = ""
                    continue

                state.silent_for_sec = 0.0
                infer_started = time.perf_counter()
                text = self._transcribe(samples)
                infer_sec = time.perf_counter() - infer_started
                self._debug_audio(rms, audio_sec, raw_text=text)
                if text:
                    self._emit_output(state, text, audio_sec, infer_sec)
        finally:
            self.running = False
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            if process.stderr:
                process.stderr.close()
            if process.stdout:
                process.stdout.close()

    def _handle_stop(self, signum, _frame):
        del signum
        self.running = False


def main():
    args = parse_args()
    if args.list_sources:
        for line in list_sources():
            print(line)
        return

    preload_cuda_runtime_libs(args.device)
    app = SystemAudioChineseTranslator(args)
    app.run()


if __name__ == "__main__":
    main()
