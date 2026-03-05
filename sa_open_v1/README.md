# Stable Audio Open v1 - Worklog

## Branch: `stable_audio_open_v1`

---

## Baseline Run

**Date:** 2026-03-02

**Model:** `stabilityai/stable-audio-open-1.0`
**Hardware:** RunPod (NVIDIA GPU 5070, CUDA 12.8)
**Framework:** vllm-omni 0.1.dev740, vLLM 0.16.0, PyTorch 2.9.1+cu128, diffusers 0.36.0

### Command
```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a hammer hitting a wooden surface" \
  --negative-prompt "Low quality" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 100 \
  --output stable_audio_output.wav
```

### Parameters
| Parameter          | Value                                          |
|--------------------|------------------------------------------------|
| Prompt             | The sound of a hammer hitting a wooden surface |
| Negative prompt    | Low quality                                    |
| Audio length       | 10.0s                                          |
| Inference steps    | 100                                            |
| Guidance scale     | 7.0                                            |
| Seed               | 42                                             |
| Sample rate        | 44100 Hz                                       |

### Timing Results (Baseline)
| Metric                        | Value          |
|-------------------------------|----------------|
| **Total wall time**           | **~112s**      |
| **Total generation time**     | **3.96s**      |
| Model loading (weights)       | 1.90s          |
| Model loading (full)          | 6.08s          |
| Post-processing               | 0.0174s        |
| GPU memory after model load   | 3.37 GiB       |
| Model size in memory          | 2.79 GiB       |
| Attention backend             | PyTorch SDPA   |

### Notes
- First run included ~40.5s model download from HuggingFace; baseline measured on second run with cached weights.
- No Flash Attention backend found; fell back to PyTorch SDPA.
- Regional compilation skipped (model does not define `_repeated_blocks`).
- Output: `stable_audio_output.wav` (10.0s audio at 44100 Hz)

---

## Worklog

| # | Change Description | Generation Time | Model Loading Time | Total Wall Time | Notes |
|---|--------------------|-----------------|-------------------|-----------------|-------|
| 0 | Baseline           | 3.96s           | 6.08s             | ~112s           | Initial run, no optimizations |

## To-Do

- [ ] Port inference speed-up changes from local repo to `vllm-omni` ([local_repo](https://github.com/csk7/stable-audio-tools/tree/fused_op_v1/inference_speedup)).
