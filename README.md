# SysGPT Dataset and Benchmark
This repository contains the full dataset and evaluation benchmark introduced in
our OSDI'25 paper:

"Principles and Methodologies for Serial Performance Optimization (OSDI' 25)"

## Overview

Large language models (LLMs) hold promise as assistants for system performance
optimization, yet their evaluation in this domain remains underexplored. This
repository provides:

- A curated dataset of performance optimization problems and observations, derived from 10 years of SOSP/OSDI papers
- A taxonomy-grounded benchmark to assess LLMs' ability to suggest concrete, actionable system optimizations
- Scripts to evaluate models on their ability to recover real-world optimization strategies

## Contents

```
.
├── dataset/
│ ├── dataset.xlsx  # Full training + test data (see below)
│ ├── example_3     # Few-shot prompt examples (N = 3)
│ ├── example_5     # Few-shot prompt examples (N = 5)
│ └── example_10    # Few-shot prompt examples (N = 10)
│
├── eval.py         # Evaluation script (e.g., precision/recall)
├── run_test.sh     # Script to reproduce Figure 7
└── README.md

```

### `dataset.xlsx`

- Sheet 1: Training dataset distilled from 10 years of OSDI/SOSP papers (2013–2022).
- Sheet 2: Test dataset of 96 papers published in 2024 (OSDI/SOSP).
- Each entry includes a problem statement, system observations, and labeled methodologies.


## Citation
If you use this dataset or benchmark, please cite:

    ```
    @inproceedings{park:sysgpt,
      title        = {{Principles and Methodologies for Serial Performance Optimization}},
      author       = {Sujin Park and Mingyu Guan and Xiang Cheng and Taesoo Kim},
      booktitle    = {Proceedings of the 19th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
      month        = jul,
      year         = 2025,
    }
    ```
