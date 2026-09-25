CUDA_VISIBLE_DEVICES=0,1 python blend_hotpotqa_freq_qwen.py --model-path Qwen2.5-32B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager

# Optional spectral method: append --method spectral to the command above.
# See README for fixed ratios, GSS calibration, and disk-cache options.
