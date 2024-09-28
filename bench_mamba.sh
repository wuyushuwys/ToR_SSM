# Peak memory benchmark
python src/mamba_bench.py --model mamba \
    --model_args "pretrained=state-spaces/mamba2-2.7b" \
    --tasks arc_easy \
    --device cuda \
    --batch_size 96
