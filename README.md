# Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs

<a href='https://arxiv.org/abs/2502.19041'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>

![method](./assets/method.png)

![case](./assets/case.png)

## Abstract

Although Aligned Large Language Models (LLMs) are trained to reject harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essences" remain the same. To address this issue, we introduce EDDF, an Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20%, underscoring its superior robustness against jailbreak attacks.

## Quick Start

1. Configure relevant parameters in config.py
2. Offline Essence Database Construction

```
cd EDDF
python offline_essense_extraction.py
python offine_essense_judge.py
python vectorstore.py
```

2. Online Adversarial Query Detection

```
python online_main.py
```

## Contact

- Shiyu Xiang: xiangshiyu@stu.scu.edu.cn
- Ronghao Chen: chenronghao@stu.pku.edu.cn

## BibTeX:

```
@article{xiang2025beyond,
  title={Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs},
  author={Xiang, Shiyu and Zhang, Ansen and Cao, Yanfei and Fan, Yang and Chen, Ronghao},
  journal={arXiv preprint arXiv:2502.19041},
  year={2025}
}
```