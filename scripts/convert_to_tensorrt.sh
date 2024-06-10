echo $PWD

MAX_JOBS=2 python3 TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir "models/mistral-7b/models--mistralai--Mistral-7B-Instruct-v0.2" \
    --output_dir "tensorrt_models/Mistral-7B-Instruct-v0.2" \
    --dtype float16

trtllm-build --checkpoint_dir tensorrt_models/Mistral-7B-Instruct-v0.2 \
            --output_dir tensorrt_models/Mistral-7B-Instruct-v0.2-tensorrt \
            --gemm_plugin float16 \
            --max_input_len 32768