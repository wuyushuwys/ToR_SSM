# Rethinking Token Reduction for State Space Models

Official Implementation of **EMNLP2024** Rethinking Token Reduction for State Space Models

> **Rethinking Token Reduction for State Space Models**   
> [Zheng Zhan](https://zhanzheng8585.github.io/), [Yushu Wu](https://scholar.google.com/citations?user=3hEDsFYAAAAJ&hl=en), Zhenglun Kong, Changdi Yang, Yifan Gong, Xuan Shen, Xue Lin, and Yanzhi Wang
> Northeastern University  
> The 2024 Conference on Empirical Methods in Natural Language Processing ([**EMNLP2024**](https://2024.emnlp.org/))


## Dependencies
```bash
pip install -r requirements.txt
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
pip install lm-eval==0.4.2
```


## Evaluation
- Please refer to `evaluate_mamba.sh` for evaluation.
- Please refer to `bench_mamba.sh` for benchmarking the peak memory.
- For config related to mamba, please follow [Mamba-ssm](https://github.com/state-spaces/mamba).
- For more detail, please follow Sec.5 in the paper.