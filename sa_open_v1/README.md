# Stable Audio Open v1
---

## Baseline Run

**Model:** `stabilityai/stable-audio-open-1.0`
**Hardware:** RunPod (NVIDIA GPU RTX A6000 Ada CUDA 12.8)
**Framework:** vllm-omni 0.1.dev740, vLLM 0.16.0, PyTorch 2.9.1+cu128, diffusers 0.36.0

### Command
```bash
python profiling.py
python profiling.py --speedup
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

### Goals
Reduce latency of memory bound operations in Stability audio open models using fusion ops. Baseline latency is 5.3 sec fot 47 sec audio generation. Memory bound ops take about 0.8 secs, Matrix multiply and flash attention take about 85% of the timeleine (4.5 sec). Reduced the 0.8 secs latency in memory bound ops to 0.4 secs (about 50% of memory bound ops latency). The original precision of the ops are maintained.

Detailed explanation of each fusion ops and Cross attention KV caching - ([local_repo](https://github.com/csk7/stable-audio-tools/tree/fused_op_v1/inference_speedup))


### Results

| # | Change Description | Generation Time | Model Loading Time | Total Wall Time | Notes |
|---|--------------------|-----------------|-------------------|-----------------|-------|
| 0 | Baseline           | 5.3s           | 6.08s             | ~112s           | Initial run, no optimizations |
| 1 | Fused ops + Prompt KV Cache           | 4.9s           | 6.08s             | ~111s           | All optimizations in  ([local_repo](https://github.com/csk7/stable-audio-tools/tree/fused_op_v1/inference_speedup))|



### Future Work - Reduce MM and FA latency for higher overall latency reduction:
1. Distallation to reduce number of diffusion steps
2. LoRA to reduce parameters
3. Quantization to FP8, NVFP4

