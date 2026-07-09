# GLARE: Geometry-based Local Anomaly Recognition and Encoding

**Few-Shot 3D Industrial Anomaly Detection from Point Cloud Geometry Alone**

GLARE is a **training-free, depth-only** method for few-shot 3D industrial anomaly
detection. It shows that neither paired RGB-D sensors nor large pretrained encoders
are fundamental to competitive 3D anomaly detection: a compact **12D hand-crafted
geometric descriptor** combined with a **density-aware memory bank** matches or
approaches full-shot RGB-D baselines while running orders of magnitude faster.

> Paper: *GLARE: Geometry-based Local Anomaly Recognition and Encoding for Few-Shot
> 3D Industrial Anomaly Detection* (under review, AAAI).

---

## Highlights

| Benchmark | Protocol | GLARE AUROC |
|-----------|----------|-------------|
| Real3D-AD | 8-shot, depth-only | **66.1%** (95% CI [65.7, 66.4], 10 seeds) |
| MVTec3D-AD | 8-shot, depth-only | **75.1%** |
| MVTec3D-AD (geometric subset, 7 cats) | 8-shot, depth-only | **76.2%** (vs. M3DM 70.4% overall) |

- **12D geometric descriptor** — 7D eigenvalue-ratio shape features + 5D local
  height statistics, selected by ablating **7,000+ configurations**. Distance
  features, common in prior work, consistently *hurt* and are excluded by design.
- **Single-scale is enough** — a single neighborhood scale `k=16` matches a
  five-scale ensemble at **4× lower latency**.
- **Density-Aware Memory Scoring (DAMS)** — a parameter-free percentile-weighted
  pooling operator that corrects for variable point density, giving a consistent
  **+2.2 pp** over standard max-pooling.
- **Fast** — **31 ms** per sample, **26× faster than FPFH**.

---

## Repository layout

```
.
├── glare/                       # Reference implementation
│   ├── glare_dams.py            # Main pipeline: 12D features + DAMS + multi-seed eval
│   ├── glare_core.py            # GPU-accelerated core (feature extraction + memory bank)
│   └── glare_pro.py             # Extended / configurable variant
├── scripts/                     # Reproduce paper experiments
│   ├── run_full_benchmark.py    # Full Real3D-AD + MVTec3D-AD benchmark
│   ├── ablation_feature_groups.py  # Feature-group ablation (Table 3)
│   ├── baseline_comparison.py   # FPFH / classical geometric baselines
│   └── glare_plus.py            # GLARE+ extended experiments
├── results/                     # Raw JSON result summaries used in the paper
├── figures/                     # Paper figures (per-category comparison, ablations)
├── requirements.txt
├── LICENSE                      # MIT
└── README.md
```

## Installation

```bash
git clone <this-repo> glare && cd glare
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GLARE is training-free and CPU-runnable; a CUDA GPU accelerates neighborhood
feature computation. FAISS is optional (the code falls back to scikit-learn's
KDTree for memory-bank nearest-neighbor scoring).

## Datasets

Download the public benchmarks and point the loaders at them:

- **Real3D-AD** — https://github.com/M-3LAB/Real3D-AD
- **MVTec 3D-AD** — https://www.mvtec.com/company/research/datasets/mvtec-3d-ad

Expected layout (depth maps are read as TIFF):

```
Real3D-mvtec/<category>/{train,test}/...
MVTec3d/<category>/{train,test,validation}/...
```

Edit the dataset root paths at the top of the scripts in `glare/` and `scripts/`
before running.

## Quick start

Run the main GLARE pipeline (12D descriptor + DAMS) with multi-seed validation:

```bash
python glare/glare_dams.py
```

Reproduce the full benchmark:

```bash
python scripts/run_full_benchmark.py
```

Reproduce the feature-group ablation:

```bash
python scripts/ablation_feature_groups.py
```


## Method in one paragraph

For each point we build a local neighborhood (`k=16`), compute the covariance
eigenvalues and derive **7 scale-invariant shape ratios** (linearity, planarity,
sphericity, anisotropy, omnivariance, curvature-like ratios) plus **5 local
height statistics** relative to the neighborhood — a **12D descriptor** per point.
Few-shot "normal" descriptors form a coreset memory bank. Anomaly scores are the
nearest-neighbor distance to this bank, pooled per sample with **DAMS**, a
density-aware percentile-weighted operator that down-weights over-dense regions
so genuine geometric deviations are not washed out.



## License

Released under the [MIT License](LICENSE).
