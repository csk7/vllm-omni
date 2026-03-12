# Stable Audio Open v1 - Worklog

## Branch: `stable_audio_open_v1`

---

## Baseline Run

**Model:** `stabilityai/stable-audio-open-1.0`
**Hardware:** RunPod (NVIDIA GPU RTX A6000 Ada CUDA 12.8)
**Framework:** vllm-omni 0.1.dev740, vLLM 0.16.0, PyTorch 2.9.1+cu128, diffusers 0.36.0

### Command
```bash
python profiling.py
python profling.py --speedup
```

### Parameters
| Parameter          | Value                                          |
|--------------------|------------------------------------------------|
| Prompt             | The sound of a hammer hitting a wooden surface |
| Negative prompt    | Low quality                                    |
| Audio length       | 47.0s                                          |
| Inference steps    | 100                                            |
| Guidance scale     | 7.0                                            |
| Seed               | 42                                             |
| Sample rate        | 44100 Hz                                       |


### Results

| # | Change Description | Generation Time | Model Loading Time | Total Wall Time | Notes |
|---|--------------------|-----------------|-------------------|-----------------|-------|
| 0 | Baseline           | 5.3s           | 6.08s             | ~112s           | Initial run, no optimizations |
| 1 | Fused ops + Prompt KV Cache           | 4.8s           | 6.08s             | ~111s           | All optimizations in  ([local_repo](https://github.com/csk7/stable-audio-tools/tree/fused_op_v1/inference_speedup))|

