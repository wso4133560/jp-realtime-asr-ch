# Japanese Realtime ASR to Chinese Translation

Run Japanese speech recognition locally on NVIDIA GPU and turn live system audio into Simplified Chinese in realtime.

Built with `ReazonSpeech K2` for Japanese ASR and `Ollama` for local translation, this repo is designed for demos, monitoring, bilingual workflows, and fast internal deployment.

## Highlights

- Local Japanese ASR on CUDA.
- Realtime capture from PulseAudio monitor sources.
- Japanese to Simplified Chinese translation through Ollama.
- Simple shell entrypoints for benchmark and live demo.
- Tested locally on RTX 3080 with more than `120x` realtime speed on the sample file.

## CPU-Only ASR

You can run the project in ASR-only mode on CPU without starting Ollama.

Offline token/s benchmark:

```bash
./run_benchmark_asr_cpu.sh --audio demo.mp3 --precision fp32 --language ja --cpu-threads 4 --warmup-runs 1 --runs 3
```

Realtime ASR only:

```bash
./run_system_audio_asr_cpu.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --cpu-threads 4 \
  --duration-sec 15
```

The realtime script prints per-decode `tokens=` and `tokps=` and ends with cumulative `asr_avg_tokens_per_second=` statistics. For CPU, `--cpu-threads 0` means auto, which currently picks up to 4 threads for a single ASR instance.

CPU-only environment setup:

```bash
python -m venv venv
venv/bin/pip install -U pip setuptools wheel numpy sherpa-onnx
venv/bin/pip install git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/k2-asr
```

Note: the checked-in environment setup below is CUDA-oriented. For a true CPU-only setup, install `sherpa-onnx` from PyPI instead of the CUDA wheel.

## Demo Snapshot

Offline benchmark command:

```bash
./run_benchmark_cuda.sh --audio demo.mp3 --device cuda --precision fp32 --language ja --warmup-runs 1 --runs 3
```

Example output:

```text
audio_seconds=17.000
model_load_seconds=4.835
avg_latency_seconds=0.141
avg_realtime_speedup=120.713
last_text=長野県は全国で三番目に大きな県ですお隣の山梨県の三倍以上もあります長野から飯田へ行くのにも東京へ行くのと同じ時間がかかるのを見ても面積の広いことが分かります
```

## Quick Start

### 1. Create the Environment

```bash
python -m venv venv
venv/bin/pip install -U pip setuptools wheel
venv/bin/pip install https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/1.12.31/sherpa_onnx-1.12.31+cuda12.cudnn9-cp310-cp310-linux_x86_64.whl
venv/bin/pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
venv/bin/pip install git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/k2-asr
curl -fsSL -o demo.mp3 https://research.reazon.jp/_static/demo.mp3
```

If you already have the upstream repository checked out locally, you can replace the Git install with:

```bash
venv/bin/pip install ./upstream/pkg/k2-asr
```

### 2. Verify Offline Japanese ASR

CUDA:

```bash
./run_benchmark_cuda.sh --audio demo.mp3 --device cuda --precision fp32 --language ja --warmup-runs 1 --runs 3
```

CPU:

```bash
./run_benchmark_asr_cpu.sh --audio demo.mp3 --precision fp32 --language ja --cpu-threads 4 --warmup-runs 1 --runs 3
```

### 3. Start Realtime Translation

Start Ollama in a separate terminal:

```bash
ollama serve
```

Pull the default translation model:

```bash
ollama pull translategemma:4b-it-q4_K_M
```

List available PulseAudio sources:

```bash
./run_system_audio_translate_reazon.sh --list-sources
```

Then run the live pipeline:

```bash
./run_system_audio_translate_reazon.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --print-source
```

Typical startup output:

```text
status=loading_asr_model
status=asr_model_ready
capture_source=alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor
device=cuda
translate_model=translategemma:4b-it-q4_K_M
ready=true
```

After `ready=true`, the script emits Japanese ASR lines and Chinese translation lines.

## Best-Fit Use Cases

- Livestream monitoring for Japanese content.
- Internal meetings for Japanese and Chinese speaking teams.
- Subtitle drafting for podcasts, videos, and event recordings.
- Booth demos for local speech AI on consumer GPUs.
- QA review for bilingual audio workflows.

## How It Works

```text
System Audio
  -> PulseAudio monitor source
  -> parec capture
  -> ReazonSpeech K2 ASR
  -> stable Japanese text
  -> Ollama /api/generate
  -> Simplified Chinese translation
```

The runtime filters unstable short fragments before sending text to Ollama, which keeps translation output cleaner in live conditions.

## Main Commands

### Offline Benchmark

```bash
./run_benchmark_cuda.sh --audio demo.mp3 --device cuda --precision fp32 --language ja --warmup-runs 1 --runs 3
```

### Offline Benchmark on CPU

```bash
./run_benchmark_asr_cpu.sh --audio demo.mp3 --precision fp32 --language ja --cpu-threads 4 --warmup-runs 1 --runs 3
```

### Realtime Translation

```bash
./run_system_audio_translate_reazon.sh --print-source
```

### Realtime ASR on CPU Without Translation

```bash
./run_system_audio_asr_cpu.sh --cpu-threads 4 --duration-sec 15
```

### Pin a Specific Source

```bash
./run_system_audio_translate_reazon.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --print-source
```

### Debug Audio Capture

```bash
./run_system_audio_translate_reazon.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --print-source \
  --debug-audio
```

### Save Output to Text Files

```bash
./run_system_audio_translate_reazon.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --print-source \
  --output-source-txt source.txt \
  --output-txt translation.txt
```

### Run a Short Smoke Test

```bash
./run_system_audio_translate_reazon.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --print-source \
  --duration-sec 15
```

## Ollama Integration

The realtime translator uses the local Ollama HTTP API at `http://127.0.0.1:11434/api/generate`.

Quick health check:

```bash
curl -fsSL http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "translategemma:4b-it-q4_K_M",
    "prompt": "Translate into Simplified Chinese only.\n\n今日はいい天気です。",
    "stream": false,
    "options": {"temperature": 0}
  }'
```

Useful overrides:

```bash
TRANSLATE_MODEL=qwen2.5:7b-instruct ./run_system_audio_translate_reazon.sh --print-source
OLLAMA_HOST=http://127.0.0.1:11434 ./run_system_audio_translate_reazon.sh --print-source
./run_system_audio_translate_reazon.sh --ollama-timeout-sec 30 --print-source
```

## Configuration

- Default ASR device: `cuda`
- CPU override: `REAZON_DEVICE=cpu`
- CPU thread override: `REAZON_CPU_THREADS=4` or `--cpu-threads 4`
- Disable translation at runtime: `--disable-translation`
- CPU benchmark helper: `run_benchmark_asr_cpu.sh`
- CPU realtime ASR helper: `run_system_audio_asr_cpu.sh`
- Specific GPU override: `REAZON_DEVICE=cuda:0`
- Translation model override: `TRANSLATE_MODEL=...`
- Ollama host override: `OLLAMA_HOST=http://host:11434`
- The translator uses `temperature=0` for stable output.
- Mixed-script translation output is retried with a stricter Chinese-only prompt.

## Requirements

- Linux with PulseAudio or PipeWire compatibility for `pactl` and `parec`
- NVIDIA GPU for the intended CUDA path, or CPU for the ASR-only path
- Python 3.10
- Ollama only when translation is enabled

## FAQ / Troubleshooting

### `venv/bin/python` or `venv` does not exist

Create the environment first:

```bash
python -m venv venv
```

Then rerun the install steps from `Quick Start`.

### `./upstream/pkg/k2-asr` does not exist

This repository does not require a local upstream checkout.
Use the Git-based install command from `Quick Start`:

```bash
venv/bin/pip install git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/k2-asr
```

### The realtime script shows no source output

Check these in order:

- Confirm the monitor source with `./run_system_audio_translate_reazon.sh --list-sources`.
- Make sure playback starts after the script prints `ready=true`.
- Try pinning the source explicitly with `--source ...monitor`.
- Use `--debug-audio` to inspect capture RMS and raw ASR behavior.

### Ollama translation is not working

Check these in order:

- Start the server with `ollama serve`.
- Confirm the model is present with `ollama list`.
- Pull the default model with `ollama pull translategemma:4b-it-q4_K_M`.
- Run the HTTP health check in `Ollama Integration`.
- If Ollama is remote, set `OLLAMA_HOST=http://host:11434`.

### CUDA startup fails

Check these in order:

- Confirm `nvidia-smi` works on the host.
- Reinstall the CUDA runtime Python packages listed in `Quick Start`.
- Rerun the offline benchmark first to isolate ASR issues from realtime capture issues.
- If needed, force CPU mode with `REAZON_DEVICE=cpu` to verify the rest of the pipeline.

### Translation quality looks unstable on short fragments

This is expected for very short or noisy live segments. Try:

- Increasing source stability by using cleaner playback.
- Running with `--print-source` to compare ASR against translation.
- Extending the runtime window through the existing script options if you are tuning for a specific environment.

## Roadmap

- Add optional subtitle-friendly output formats such as `.srt` or `.vtt`.
- Add richer README assets such as screenshots, terminal captures, or architecture diagrams.
- Add support for more translation model presets and recommended quality/speed tradeoffs.
- Add a reproducible demo script for end-to-end local showcases.
- Add containerized setup for faster onboarding on clean Linux machines.

## Repository Layout

- `benchmark_reazonspeech_k2_cuda.py`: offline benchmark runner
- `reazon_asr_runtime.py`: local model loader with configurable CPU threads
- `run_benchmark_cuda.sh`: benchmark wrapper with CUDA library paths
- `run_benchmark_asr_cpu.sh`: CPU benchmark wrapper
- `system_audio_translate_reazon.py`: realtime ASR and translation entrypoint
- `run_system_audio_translate_reazon.sh`: realtime wrapper script
- `run_system_audio_asr_cpu.sh`: CPU realtime ASR-only wrapper
- `demo.mp3`: sample Japanese audio for validation

## Main Entrypoint

```text
system_audio_translate_reazon.py
```
