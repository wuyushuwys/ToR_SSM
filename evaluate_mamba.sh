# mamba2-2.7b performance benchmark
python src/mamba_eval.py --model mamba \
    --model_args "pretrained=state-spaces/mamba2-2.7b" \
    --tasks lambada_openai,arc_challenge,arc_easy,piqa,winogrande,hellaswag \
    --device cuda \
    --batch_size 96