# ReazonSpeech K2 CUDA Benchmark

## Environment

```bash
python -m venv venv
venv/bin/pip install -U pip setuptools wheel
venv/bin/pip install https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/1.12.31/sherpa_onnx-1.12.31+cuda12.cudnn9-cp310-cp310-linux_x86_64.whl
venv/bin/pip install ./upstream/pkg/k2-asr
venv/bin/pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
curl -fsSL -o demo.mp3 https://research.reazon.jp/_static/demo.mp3
```

## Benchmark

```bash
./run_benchmark_cuda.sh --audio demo.mp3 --device cuda --precision fp32 --language ja --warmup-runs 1 --runs 3
```

The wrapper script exports the NVIDIA runtime library paths needed by the `sherpa-onnx` CUDA wheel.

## System Audio Translation

The realtime system-audio translator captures PulseAudio monitor audio, transcribes it with ReazonSpeech K2 on CUDA, and translates the Japanese speech into Simplified Chinese with `translategemma:4b-it-q4_K_M`.

### Run Script

```bash
./run_system_audio_translate_reazon.sh --print-source
```

### List Sources

```bash
./run_system_audio_translate_reazon.sh --list-sources
```

### Pin A Source

```bash
./run_system_audio_translate_reazon.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --print-source
```

### Debug Mode

```bash
./run_system_audio_translate_reazon.sh \
  --source alsa_output.pci-0000_04_00.1.hdmi-stereo.monitor \
  --print-source \
  --debug-audio
```

### Notes

- The run script uses `venv/bin/python` directly and defaults to `--device cuda`.
- Override the device with `REAZON_DEVICE=cpu` or `REAZON_DEVICE=cuda:0`.
- Override the translation model with `TRANSLATE_MODEL=...`.
- The translator suppresses many unstable short fragments and retries mixed-script outputs with a stricter Chinese-only prompt.
- Current main entrypoint: `system_audio_translate_reazon.py`
