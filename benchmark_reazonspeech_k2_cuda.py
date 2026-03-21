#!/usr/bin/env python3

import argparse
import statistics
import time

from reazonspeech.k2.asr import audio_from_path, load_model, transcribe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark reazonspeech-k2-v2 on CUDA and report tokens/s."
    )
    parser.add_argument("--audio", required=True, help="Path to an input audio file.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "coreml"])
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "int8", "int8-fp32"],
        help="Model precision to use from Hugging Face.",
    )
    parser.add_argument(
        "--language",
        default="ja",
        choices=["ja", "ja-en", "ja-en-mls-5k"],
        help="Which ReazonSpeech K2 model to load.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup decode runs before timing.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Timed decode runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    audio = audio_from_path(args.audio)
    audio_seconds = len(audio.waveform) / audio.samplerate

    load_started = time.perf_counter()
    model = load_model(
        device=args.device,
        precision=args.precision,
        language=args.language,
    )
    load_elapsed = time.perf_counter() - load_started

    warmup_result = None
    for _ in range(args.warmup_runs):
        warmup_result = transcribe(model, audio)

    latencies = []
    token_counts = []
    texts = []
    for _ in range(args.runs):
        started = time.perf_counter()
        result = transcribe(model, audio)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        token_counts.append(len(result.subwords))
        texts.append(result.text)

    total_tokens = sum(token_counts)
    total_time = sum(latencies)
    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(token_counts)
    avg_tokens_per_second = total_tokens / total_time if total_time else 0.0
    realtime_factor = avg_latency / audio_seconds if audio_seconds else 0.0
    realtime_speedup = audio_seconds / avg_latency if avg_latency else 0.0

    print(f"audio_path={args.audio}")
    print(f"audio_seconds={audio_seconds:.3f}")
    print(f"device_requested={args.device}")
    print(f"precision={args.precision}")
    print(f"language={args.language}")
    print(f"model_load_seconds={load_elapsed:.3f}")
    print(f"warmup_runs={args.warmup_runs}")
    print(f"timed_runs={args.runs}")
    print(f"avg_latency_seconds={avg_latency:.3f}")
    print(f"min_latency_seconds={min(latencies):.3f}")
    print(f"max_latency_seconds={max(latencies):.3f}")
    print(f"avg_tokens={avg_tokens:.1f}")
    print(f"avg_tokens_per_second={avg_tokens_per_second:.3f}")
    print(f"avg_realtime_factor={realtime_factor:.3f}")
    print(f"avg_realtime_speedup={realtime_speedup:.3f}")
    if warmup_result is not None:
        print(f"warmup_text={warmup_result.text}")
    print(f"last_text={texts[-1]}")


if __name__ == "__main__":
    main()
