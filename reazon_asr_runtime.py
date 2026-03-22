#!/usr/bin/env python3
import os

import huggingface_hub as hf
import sherpa_onnx

MODEL_SPECS = {
    "ja": {
        "repo_id": "reazon-research/reazonspeech-k2-v2",
        "epochs": 99,
    },
    "ja-en": {
        "repo_id": "reazon-research/reazonspeech-k2-v2-ja-en",
        "epochs": 35,
    },
    "ja-en-mls-5k": {
        "repo_id": "reazon-research/reazonspeech-k2-v2-ja-en-mls-5k-corrected",
        "epochs": 21,
    },
}


def normalize_provider(device):
    if device.startswith("cuda"):
        return "cuda"
    return device


def resolve_cpu_threads(device, cpu_threads=0):
    provider = normalize_provider(device)
    if provider != "cpu":
        return 1
    if cpu_threads < 0:
        raise ValueError("cpu_threads must be >= 0")
    if cpu_threads > 0:
        return cpu_threads
    return max(1, min(os.cpu_count() or 1, 4))


def load_model(device="cpu", precision="fp32", language="ja", cpu_threads=0):
    if language not in MODEL_SPECS:
        raise ValueError(f"Unknown language: {language!r}")

    spec = MODEL_SPECS[language]
    epochs = spec["epochs"]
    files_by_precision = {
        "fp32": {
            "tokens": "tokens.txt",
            "encoder": f"encoder-epoch-{epochs}-avg-1.onnx",
            "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
            "joiner": f"joiner-epoch-{epochs}-avg-1.onnx",
        },
        "int8": {
            "tokens": "tokens.txt",
            "encoder": f"encoder-epoch-{epochs}-avg-1.int8.onnx",
            "decoder": f"decoder-epoch-{epochs}-avg-1.int8.onnx",
            "joiner": f"joiner-epoch-{epochs}-avg-1.int8.onnx",
        },
        "int8-fp32": {
            "tokens": "tokens.txt",
            "encoder": f"encoder-epoch-{epochs}-avg-1.int8.onnx",
            "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
            "joiner": f"joiner-epoch-{epochs}-avg-1.int8.onnx",
        },
    }

    if precision not in files_by_precision:
        raise ValueError(f"Unknown precision: {precision!r}")

    try:
        basedir = hf.snapshot_download(spec["repo_id"], local_files_only=True)
    except hf.utils.LocalEntryNotFoundError:
        basedir = hf.snapshot_download(spec["repo_id"])

    files = files_by_precision[precision]
    provider = normalize_provider(device)
    num_threads = resolve_cpu_threads(provider, cpu_threads)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=os.path.join(basedir, files["tokens"]),
        encoder=os.path.join(basedir, files["encoder"]),
        decoder=os.path.join(basedir, files["decoder"]),
        joiner=os.path.join(basedir, files["joiner"]),
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=provider,
    )
