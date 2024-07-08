## Scoring pretraining corpora with LLMs

Testing scoring Swedish/Norwegian pretraining corpora with Mistral.

### Installation

```bash
# May need to install flash-attn manually if it fails to install
MAX_JOBS=6 pip install -r requirements.txt
pip install --editable .
```

### Huggingface inference

See 

* `askllm_hf.py` for Ask-LLM style inference with Huggingface.
* `fineweb_hf_llama.py` for Fineweb-Edu style document scoring with Llama3-70b-instruct.
* `fineweb_hf_gemma.py` for Fineweb-Edu style document scoring with Gemma2-27b-it.

### TensorRT-LLM inference

Build a singularity container with TensorRT-LLM installed:

```bash
singularity build trt-llm.sif containers/tensorrt.def --fakeroot
```

Run the container interactively:

```bash
singularity shell --nv trt-llm.sif
```

Or, alternatively, follow the [instructions in the TensorRT-LLM repo to run with docker](https://github.com/NVIDIA/TensorRT-LLM/tree/main?tab=readme-ov-file#installation).

#### Download and convert models

If you have access to internet on your nodes: 

```bash
scripts/convert_to_tensorrt.py
```

Otherwise, first download a model checkpoint on a login node: 

```bash
python3 scripts/download_model_checkpoint.py
```

### Lessons learned

* Quantized models consume less memory but don't necessarily have higher throughput (tokens/second).
* torch.compile doesn't seem to provide a significant speedup over flash attention 2.

### TODO

* [ ] Benchmark throughput **HF** vs **HF with torch.compile** vs **TRT-LLM**.
* [ ] Benchmark throughput with different pretraining document text lengths (truncate).
* [ ] Get TensorRT-LLM to work on HPC nodes.

### Benchmarks

Scoring 500k documents with Fineweb-Edu style scoring on A100s with 65GB VRAM:

* 560 A100 GPU hours for Llama3-70b-instruct.
* 280 A100 GPU hours for Gemma2-27b-it.