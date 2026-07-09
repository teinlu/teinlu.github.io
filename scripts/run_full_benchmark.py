import os
import sys
import json
import time
import traceback
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import tifffile
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────
# GPU Setup
# ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] ✓ {gpu_name}  |  {total_mem_gb:.1f} GB VRAM")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Limit single-op memory to 80% of VRAM
    VRAM_LIMIT_GB = total_mem_gb * 0.80
else:
    print("[WARNING] CUDA not available — CPU fallback (slow)")
    VRAM_LIMIT_GB = 8.0

# ─────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────
REAL3D_ROOT  = Path('/Real3D')
MVTEC3D_ROOT = Path('/MVTec3d')
RESULTS_DIR  = Path('/results')
RESULTS_DIR.mkdir(exist_ok=True)


def load_tiff_points(file_path: str, n_points: int = 2048,
                     seed: int = 42) -> Optional[np.ndarray]:
    """Load TIFF depth map → normalized point cloud (N, 3) float32."""
    try:
        data = tifffile.imread(str(file_path))
        if data is None:
            return None

        if data.ndim == 3 and data.shape[2] == 3:
            pts = data.reshape(-1, 3).astype(np.float32)
        elif data.ndim == 3 and data.shape[0] == 3:
            pts = np.transpose(data, (1, 2, 0)).reshape(-1, 3).astype(np.float32)
        elif data.ndim == 2:
            valid = np.isfinite(data) & (data > 0)
            y_i, x_i = np.where(valid)
            pts = np.stack([x_i, y_i, data[valid]], axis=1).astype(np.float32)
        else:
            return None

        valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
        pts = pts[valid]
        if len(pts) < 50:
            return None

        # Normalize to unit sphere
        center = pts.mean(axis=0)
        pts -= center
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts /= scale

        # Random subsampling (GPU FPS too slow for CPU pre-loading)
        if len(pts) > n_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pts), n_points, replace=False)
            pts = pts[idx]

        return pts
    except Exception:
        return None


def fps_gpu(pts: torch.Tensor, n_points: int) -> torch.Tensor:
    """Farthest Point Sampling on GPU. pts: (N, 3) → (n_points, 3)."""
    N = pts.shape[0]
    if N <= n_points:
        return pts
    selected = torch.zeros(n_points, dtype=torch.long, device=pts.device)
    dists = torch.full((N,), float('inf'), device=pts.device)
    # Random first point
    selected[0] = torch.randint(0, N, (1,)).item()
    for i in range(1, n_points):
        last = pts[selected[i - 1]].unsqueeze(0)   # (1, 3)
        d = ((pts - last) ** 2).sum(dim=1)           # (N,)
        dists = torch.minimum(dists, d)
        selected[i] = dists.argmax()
    return pts[selected]


def rotate_points_gpu(pts: torch.Tensor, angle_deg: float,
                      axis: str = 'x', seed: int = 0) -> torch.Tensor:
    """Random rotation ±angle_deg around given axis (GPU tensor)."""
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-angle_deg, angle_deg)) * np.pi / 180.0
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        R = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 'y':
        R = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    R_t = torch.tensor(R, dtype=torch.float32, device=pts.device)
    return pts @ R_t.T


# ─────────────────────────────────────────────────────
# GPU Feature Extraction (15D per scale)
# ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_features_gpu(pts: torch.Tensor, k: int = 8,
                          chunk_size: int = 4096) -> torch.Tensor:
    """
    15D geometric features per point (GPU).
    Uses chunked cdist to avoid OOM.
    pts: (N, 3) → (N, 15)
    """
    N = pts.shape[0]
    k_actual = min(k, N - 1)
    eps = 1e-8

    # ── Chunked kNN ──
    knn_idx_list, knn_dist_list = [], []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sub = pts[start:end]                         # (B, 3)
        d = torch.cdist(sub, pts)                    # (B, N)
        top_dists, top_idx = torch.topk(d, k_actual + 1, largest=False)
        knn_idx_list.append(top_idx[:, 1:])          # exclude self
        knn_dist_list.append(torch.sqrt(top_dists[:, 1:].clamp(min=0)))
    knn_idx   = torch.cat(knn_idx_list,  dim=0)     # (N, k)
    knn_dists = torch.cat(knn_dist_list, dim=0)     # (N, k)

    # Neighbor coords
    neighbors = pts[knn_idx]                         # (N, k, 3)
    centered  = neighbors - pts.unsqueeze(1)         # (N, k, 3)

    # Covariance
    cov = torch.einsum('nki,nkj->nij', centered, centered) / k_actual  # (N,3,3)

    # Eigen-decomposition
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues  = eigenvalues.flip(dims=[1]).clamp(min=1e-10)   # descending
        eigenvectors = eigenvectors.flip(dims=[2])                   # (N,3,3)
        normals      = eigenvectors[:, :, 2]                         # (N, 3)
    except Exception:
        eigenvalues = torch.ones(N, 3, device=pts.device) / 3.0
        normals     = torch.zeros(N, 3, device=pts.device)
        normals[:, 2] = 1.0

    l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    l_sum = l1 + l2 + l3 + eps

    # ── Shape (7D) ──
    linearity    = (l1 - l2) / (l1 + eps)
    planarity    = (l2 - l3) / (l1 + eps)
    sphericity   = l3 / (l1 + eps)
    anisotropy   = (l1 - l3) / (l1 + eps)
    omnivariance = (l1 * l2 * l3 + 1e-30).pow(1.0 / 3.0)
    p1, p2, p3   = l1 / l_sum, l2 / l_sum, l3 / l_sum
    eigenentropy = -(p1 * (p1 + eps).log() + p2 * (p2 + eps).log() +
                     p3 * (p3 + eps).log())
    curvature    = l3 / l_sum

    # ── Height (5D) ──
    z_neigh = neighbors[:, :, 2]
    z_std   = z_neigh.std(dim=1)
    z_range = z_neigh.max(dim=1).values - z_neigh.min(dim=1).values
    z_mean  = z_neigh.mean(dim=1)
    z_pos   = pts[:, 2]
    z_skew  = ((z_neigh - z_mean.unsqueeze(1)) ** 3).mean(dim=1) / (z_std ** 3 + eps)

    # ── Normal consistency (1D) ──
    neighbor_normals   = normals[knn_idx]              # (N, k, 3)
    normal_consistency = (normals.unsqueeze(1) * neighbor_normals).sum(dim=2).abs().mean(dim=1)

    # ── Density ratio (1D) ──
    local_density = 1.0 / (knn_dists.mean(dim=1) + eps)
    density_ratio = local_density / (local_density.mean() + eps)

    # ── Distance CV (1D) ──
    dist_cv = knn_dists.std(dim=1) / (knn_dists.mean(dim=1) + eps)

    return torch.stack([
        linearity, planarity, sphericity, anisotropy, omnivariance,
        eigenentropy, curvature,
        z_std, z_range, z_mean, z_pos, z_skew,
        normal_consistency, density_ratio, dist_cv
    ], dim=1)                                          # (N, 15)


@torch.no_grad()
def extract_multiscale_features_gpu(pts: torch.Tensor,
                                    scales: List[int] = [4, 8, 16, 32]) -> torch.Tensor:
    """Multi-scale feature concatenation: (N, 15*S)."""
    return torch.cat([extract_features_gpu(pts, k=k) for k in scales], dim=1)


# ─────────────────────────────────────────────────────
# DAMS Scoring (GPU, chunked)
# ─────────────────────────────────────────────────────

@torch.no_grad()
def dams_score_gpu(test_feat: torch.Tensor, memory: torch.Tensor,
                   uniform: bool = False,
                   chunk_size: int = 2048) -> float:
    """
    DAMS scoring: percentile-weighted NN distance aggregation.
    test_feat: (N, D), memory: (M, D) → scalar score
    Chunked cdist to avoid OOM.
    """
    # Normalize
    feat_mean = memory.mean(dim=0, keepdim=True)
    feat_std  = memory.std(dim=0, keepdim=True).clamp(min=1e-6)
    test_n    = (test_feat - feat_mean) / feat_std
    mem_n     = (memory    - feat_mean) / feat_std

    # Chunked NN distances
    min_dists_list = []
    for start in range(0, test_n.shape[0], chunk_size):
        end = min(start + chunk_size, test_n.shape[0])
        sub = test_n[start:end]
        d   = torch.cdist(sub, mem_n)               # (B, M)
        min_dists_list.append(d.min(dim=1).values)
    min_dists = torch.cat(min_dists_list)            # (N,)

    if uniform:
        return min_dists.mean().item()

    # DAMS: weighted quantile aggregation
    p_vals  = torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99], device=min_dists.device)
    quants  = torch.quantile(min_dists, p_vals)
    weights = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.35], device=min_dists.device)
    return (quants * weights).sum().item()


# ─────────────────────────────────────────────────────
# Category Cache: pre-compute all features once
# ─────────────────────────────────────────────────────

class CategoryCache:
    """Pre-loads and featurizes all point clouds for a category."""

    def __init__(self, train_files, test_normal, test_anomaly,
                 n_points: int, scales: List[int]):
        self.train_feats   = []   # list of (N, D) tensors
        self.test_n_feats  = []   # list of (N, D) tensors (normal)
        self.test_a_feats  = []   # list of (N, D) tensors (anomaly)
        self.n_points = n_points
        self.scales   = scales

        # Load & featurize with progress
        self._load_files(train_files,  self.train_feats,  'train')
        self._load_files(test_normal,  self.test_n_feats, 'test_good')
        self._load_files(test_anomaly, self.test_a_feats, 'test_bad')

    def _load_files(self, files, out_list, tag=''):
        for i, f in enumerate(files):
            pts = load_tiff_points(str(f), n_points=self.n_points, seed=i)
            if pts is None:
                continue
            pts_t = torch.from_numpy(pts).to(device)
            feat  = extract_multiscale_features_gpu(pts_t, self.scales)
            out_list.append(feat.cpu())   # store on CPU to save VRAM

    @property
    def valid(self):
        return (len(self.train_feats) >= 1 and
                len(self.test_n_feats) + len(self.test_a_feats) >= 5 and
                len(self.test_a_feats) >= 1)


# ─────────────────────────────────────────────────────
# Single-seed runner (uses pre-cached features)
# ─────────────────────────────────────────────────────

def run_seed_from_cache(cache: CategoryCache,
                        n_shots: int, seed: int,
                        dams_uniform: bool = False,
                        rotation_deg: float = 0.0) -> Optional[float]:
    """Run one seed using pre-cached features. Fast: only memory sampling."""
    rng = np.random.default_rng(seed * 1000 + 42)
    actual_shots = min(n_shots, len(cache.train_feats))
    if actual_shots < 1:
        return None
    shot_idx = rng.choice(len(cache.train_feats), actual_shots, replace=False)

    # Build memory bank (GPU)
    ref_list = []
    for idx in shot_idx:
        feat = cache.train_feats[idx].to(device)   # (N, D)
        if rotation_deg > 0:
            # Can't retroactively rotate XYZ after featurization,
            # but for GLARE test we apply rotation at feature level:
            # shuffle height features slightly to simulate orientation change
            # (proper rotation requires re-loading raw pts — done in pose exp)
            pass
        ref_list.append(feat)
    memory = torch.cat(ref_list, dim=0)             # (M, D)

    scores, labels = [], []
    def _score(feat_cpu, label):
        feat = feat_cpu.to(device)
        s = dams_score_gpu(feat, memory, uniform=dams_uniform)
        scores.append(s)
        labels.append(label)

    for f in cache.test_n_feats:
        _score(f, 0)
    for f in cache.test_a_feats:
        _score(f, 1)

    if len(scores) < 5 or len(set(labels)) < 2:
        return None
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return None


# ─────────────────────────────────────────────────────
# Pose-robust runner (re-loads raw pts with rotation)
# ─────────────────────────────────────────────────────

def run_seed_pose(train_files, test_normal, test_anomaly,
                  n_shots: int, seed: int, n_points: int,
                  scales: List[int], rotation_deg: float = 0.0) -> Optional[float]:
    """Pose robustness: re-loads raw pts and applies rotation before featurization."""
    rng = np.random.default_rng(seed * 1000 + 42)
    actual_shots = min(n_shots, len(train_files))
    if actual_shots < 1:
        return None
    shot_idx = rng.choice(len(train_files), actual_shots, replace=False)

    ref_list = []
    for idx in shot_idx:
        pts = load_tiff_points(str(train_files[idx]), n_points=n_points, seed=seed)
        if pts is None:
            continue
        pts_t = torch.from_numpy(pts).to(device)
        if rotation_deg > 0:
            pts_t = rotate_points_gpu(pts_t, rotation_deg, 'x', seed)
        ref_list.append(extract_multiscale_features_gpu(pts_t, scales))
    if not ref_list:
        return None
    memory = torch.cat(ref_list, dim=0)

    scores, labels = [], []
    def _score(f, label, sidx):
        pts = load_tiff_points(str(f), n_points=n_points, seed=seed + sidx)
        if pts is None:
            return
        pts_t = torch.from_numpy(pts).to(device)
        if rotation_deg > 0:
            pts_t = rotate_points_gpu(pts_t, rotation_deg, 'x', seed + label + sidx)
        feat = extract_multiscale_features_gpu(pts_t, scales)
        scores.append(dams_score_gpu(feat, memory))
        labels.append(label)

    for i, f in enumerate(test_normal):
        _score(f, 0, i)
    for i, f in enumerate(test_anomaly):
        _score(f, 1, i + 10000)

    if len(scores) < 5 or len(set(labels)) < 2:
        return None
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return None


# ─────────────────────────────────────────────────────
# Category file discovery
# ─────────────────────────────────────────────────────

def get_category_files(cat_path: Path, dataset_type: str):
    if dataset_type == 'real3d':
        train_dir = cat_path / 'train'
        test_dir  = cat_path / 'test'
        train_files  = sorted(train_dir.rglob('*.tiff')) + sorted(train_dir.rglob('*.tif'))
        test_normal  = sorted((test_dir / 'good').rglob('*.tiff')) + \
                       sorted((test_dir / 'good').rglob('*.tif'))
        test_anomaly = [f for f in sorted(test_dir.rglob('*.tiff')) +
                        sorted(test_dir.rglob('*.tif')) if 'good' not in str(f)]
    else:
        train_good = cat_path / 'train' / 'good'
        test_dir   = cat_path / 'test'

        def _get(d):
            fs = sorted(d.rglob('*.tiff')) + sorted(d.rglob('*.tif'))
            return fs or sorted(d.rglob('xyz/*.tiff')) + sorted(d.rglob('xyz/*.tif'))

        train_files  = _get(train_good)
        test_normal  = _get(test_dir / 'good')
        test_anomaly = []
        for sub in test_dir.iterdir():
            if sub.is_dir() and sub.name != 'good':
                test_anomaly.extend(_get(sub))

    return train_files, test_normal, test_anomaly


# ─────────────────────────────────────────────────────
# Dataset runner (parallel categories)
# ─────────────────────────────────────────────────────

def run_dataset_cached(root: Path, dataset_type: str,
                       n_shots: int, n_seeds: int, n_points: int,
                       scales: List[int], dams_uniform: bool = False,
                       label: str = '',
                       max_workers: int = 2) -> Dict:
    """
    Run GLARE across all categories using pre-cached features.
    Categories are processed in parallel with `max_workers` threads.
    """
    categories = [d for d in sorted(root.iterdir())
                  if d.is_dir() and not d.name.startswith('.')
                  and d.name not in ('license.txt', 'readme.txt')]

    tag = label or dataset_type
    print(f"\n  [{tag}] {len(categories)} cats | "
          f"{n_shots}-shot | {n_seeds} seeds | device={device}")

    def process_category(cat):
        t0 = time.time()
        train_files, test_normal, test_anomaly = get_category_files(cat, dataset_type)
        if not train_files or len(test_normal) + len(test_anomaly) < 5 or not test_anomaly:
            return cat.name, None, 0.0

        # Build cache (featurize all files once)
        try:
            cache = CategoryCache(train_files, test_normal, test_anomaly,
                                  n_points, scales)
        except Exception as e:
            return cat.name, None, 0.0

        if not cache.valid:
            return cat.name, None, 0.0

        aurocs = []
        for seed in range(n_seeds):
            auc = run_seed_from_cache(cache, n_shots, seed,
                                      dams_uniform=dams_uniform)
            if auc is not None:
                aurocs.append(auc)

        elapsed = time.time() - t0
        if not aurocs:
            return cat.name, None, elapsed

        res = {
            'mean':    float(np.mean(aurocs)),
            'std':     float(np.std(aurocs)),
            'aurocs':  [float(a) for a in aurocs],
            'n_seeds': len(aurocs),
        }
        return cat.name, res, elapsed

    results = {}
    # Use thread pool for parallel category processing
    # (GPU tensor ops happen inside each thread; torch is thread-safe for inference)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_category, cat): cat for cat in categories}
        for fut in as_completed(futures):
            name, res, elapsed = fut.result()
            if res:
                results[name] = res
                print(f"    ✓ {name:22s}  AUROC={res['mean']:.4f}±{res['std']:.4f}"
                      f"  [{elapsed:.0f}s]")
            else:
                print(f"    ✗ {name:22s}  skipped")

    if not results:
        return {'categories': {}, 'mean_auroc': 0.0, 'std_auroc': 0.0, 'n_categories': 0}

    means = [r['mean'] for r in results.values()]
    return {
        'categories':   results,
        'mean_auroc':   float(np.mean(means)),
        'std_auroc':    float(np.std(means)),
        'n_categories': len(results),
    }


def run_dataset_pose(root: Path, dataset_type: str,
                     n_shots: int, n_seeds: int, n_points: int,
                     scales: List[int], rotation_deg: float = 0.0,
                     label: str = '') -> Dict:
    """Pose robustness: re-loads raw pts with rotation per seed."""
    categories = [d for d in sorted(root.iterdir())
                  if d.is_dir() and not d.name.startswith('.')
                  and d.name not in ('license.txt', 'readme.txt')]

    tag = label or dataset_type
    print(f"\n  [{tag}] {len(categories)} cats | {n_shots}-shot | "
          f"{n_seeds} seeds | rotation=±{rotation_deg:.0f}°")

    results = {}
    for cat in categories:
        t0 = time.time()
        train_files, test_normal, test_anomaly = get_category_files(cat, dataset_type)
        if not train_files or not test_anomaly:
            continue

        aurocs = []
        for seed in range(n_seeds):
            auc = run_seed_pose(train_files, test_normal, test_anomaly,
                                n_shots=n_shots, seed=seed,
                                n_points=n_points, scales=scales,
                                rotation_deg=rotation_deg)
            if auc is not None:
                aurocs.append(auc)

        elapsed = time.time() - t0
        if aurocs:
            results[cat.name] = {
                'mean': float(np.mean(aurocs)), 'std': float(np.std(aurocs)),
                'aurocs': [float(a) for a in aurocs], 'n_seeds': len(aurocs),
            }
            print(f"    ✓ {cat.name:22s}  AUROC={results[cat.name]['mean']:.4f}±"
                  f"{results[cat.name]['std']:.4f}  [{elapsed:.0f}s]")

    if not results:
        return {'categories': {}, 'mean_auroc': 0.0, 'std_auroc': 0.0, 'n_categories': 0}
    means = [r['mean'] for r in results.values()]
    return {'categories': results, 'mean_auroc': float(np.mean(means)),
            'std_auroc': float(np.std(means)), 'n_categories': len(results)}


# ─────────────────────────────────────────────────────
# Experiment 1 — Multi-Seed Statistical Validation
# ─────────────────────────────────────────────────────

def exp1_multiseed():
    """10 seeds on both datasets — addresses statistical power criticism."""
    print("\n" + "="*65)
    print("EXP 1: Multi-Seed Statistical Validation (10 seeds)")
    print("="*65)
    cfg = dict(n_shots=8, n_seeds=10, n_points=2048, scales=[4, 8, 16, 32])
    t0  = time.time()

    r3d = run_dataset_cached(REAL3D_ROOT,  'real3d',  label='Real3D(10seed)',  **cfg)
    mvt = run_dataset_cached(MVTEC3D_ROOT, 'mvtec3d', label='MVTec3D(10seed)', **cfg)

    elapsed = time.time() - t0
    print(f"\n  Real3D-AD   mean AUROC: {r3d['mean_auroc']:.4f} ± {r3d['std_auroc']:.4f}")
    print(f"  MVTec3D-AD  mean AUROC: {mvt['mean_auroc']:.4f} ± {mvt['std_auroc']:.4f}")

    result = {
        'experiment': 'multiseed_validation', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device), 'config': cfg, 'elapsed_s': elapsed,
        'Real3D-AD': r3d, 'MVTec3D-AD': mvt,
        'summary': {
            'real3d_mean': r3d['mean_auroc'], 'real3d_std': r3d['std_auroc'],
            'mvtec_mean':  mvt['mean_auroc'], 'mvtec_std':  mvt['std_auroc'],
        }
    }
    out = RESULTS_DIR / 'round45_multiseed.json'
    out.write_text(json.dumps(result, indent=2))
    print(f" Saved → {out}  ({elapsed/60:.1f} min)")
    return result


# ─────────────────────────────────────────────────────
# Experiment 2 — DAMS Uniform-Weight Ablation
# ─────────────────────────────────────────────────────

def exp2_dams_ablation():
    """Weighted DAMS vs Uniform DAMS — proves DAMS necessity."""
    print("\n" + "="*65)
    print("EXP 2: DAMS Weighted vs Uniform Ablation (5 seeds)")
    print("="*65)
    base_cfg = dict(n_shots=8, n_seeds=5, n_points=2048, scales=[4, 8, 16, 32])
    t0 = time.time()

    w_r3d = run_dataset_cached(REAL3D_ROOT,  'real3d',  dams_uniform=False,
                               label='Weighted-R3D', **base_cfg)
    u_r3d = run_dataset_cached(REAL3D_ROOT,  'real3d',  dams_uniform=True,
                               label='Uniform-R3D',  **base_cfg)
    w_mvt = run_dataset_cached(MVTEC3D_ROOT, 'mvtec3d', dams_uniform=False,
                               label='Weighted-MVT', **base_cfg)
    u_mvt = run_dataset_cached(MVTEC3D_ROOT, 'mvtec3d', dams_uniform=True,
                               label='Uniform-MVT',  **base_cfg)

    elapsed = time.time() - t0
    g_r3d = w_r3d['mean_auroc'] - u_r3d['mean_auroc']
    g_mvt = w_mvt['mean_auroc'] - u_mvt['mean_auroc']
    print(f"\n  DAMS gain  Real3D: {g_r3d:+.4f}   MVTec: {g_mvt:+.4f}")

    result = {
        'experiment': 'dams_ablation', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device), 'config': base_cfg, 'elapsed_s': elapsed,
        'DAMS-Weighted': {'Real3D-AD': w_r3d, 'MVTec3D-AD': w_mvt},
        'DAMS-Uniform':  {'Real3D-AD': u_r3d, 'MVTec3D-AD': u_mvt},
        'summary': {
            'weighted_r3d': w_r3d['mean_auroc'], 'uniform_r3d': u_r3d['mean_auroc'],
            'gain_r3d': g_r3d,
            'weighted_mvt': w_mvt['mean_auroc'], 'uniform_mvt': u_mvt['mean_auroc'],
            'gain_mvt': g_mvt,
        }
    }
    out = RESULTS_DIR / 'round45_dams_ablation.json'
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {out}  ({elapsed/60:.1f} min)")
    return result


# ─────────────────────────────────────────────────────
# Experiment 3 — Shot Sensitivity
# ─────────────────────────────────────────────────────

def exp3_shot_sensitivity():
    """1/2/4/8/16-shot evaluation on Real3D-AD (5 seeds each)."""
    print("\n" + "="*65)
    print("EXP 3: Shot Sensitivity (1/2/4/8/16-shot, 5 seeds, Real3D)")
    print("="*65)
    base_cfg = dict(n_seeds=5, n_points=2048, scales=[4, 8, 16, 32])
    t0 = time.time()

    shot_results = {}
    for n_shots in [1, 2, 4, 8, 16]:
        r = run_dataset_cached(REAL3D_ROOT, 'real3d', n_shots=n_shots,
                               label=f'Real3D-{n_shots}shot', **base_cfg)
        shot_results[f'{n_shots}-shot'] = r
        print(f"  {n_shots:2d}-shot: {r['mean_auroc']:.4f} ± {r['std_auroc']:.4f}")

    elapsed = time.time() - t0
    result = {
        'experiment': 'shot_sensitivity', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device), 'config': base_cfg, 'elapsed_s': elapsed,
        'results':  shot_results,
        'summary':  {k: v['mean_auroc'] for k, v in shot_results.items()},
    }
    out = RESULTS_DIR / 'round45_shot_sensitivity.json'
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {out}  ({elapsed/60:.1f} min)")
    return result


# ─────────────────────────────────────────────────────
# Experiment 4 — Pose Robustness
# ─────────────────────────────────────────────────────

def exp4_pose_robustness():
    """Test GLARE under ±0°, ±5°, ±15° rotations on Real3D-AD."""
    print("\n" + "="*65)
    print("EXP 4: Pose Robustness (±0°, ±5°, ±15° rotation)")
    print("="*65)
    pose_cfg = dict(n_shots=8, n_seeds=5, n_points=2048, scales=[4, 8, 16, 32])
    t0 = time.time()

    pose_results = {}
    for deg in [0.0, 5.0, 15.0]:
        r = run_dataset_pose(REAL3D_ROOT, 'real3d', rotation_deg=deg,
                             label=f'Real3D-rot{int(deg)}deg', **pose_cfg)
        pose_results[f'rot_{int(deg)}deg'] = r
        print(f"  ±{int(deg):2d}°: {r['mean_auroc']:.4f} ± {r['std_auroc']:.4f}")

    elapsed = time.time() - t0
    deg0  = pose_results['rot_0deg']['mean_auroc']
    result = {
        'experiment': 'pose_robustness', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device), 'config': pose_cfg, 'elapsed_s': elapsed,
        'results':  pose_results,
        'summary':  {k: v['mean_auroc'] for k, v in pose_results.items()},
        'degradation_5deg':  deg0 - pose_results['rot_5deg']['mean_auroc'],
        'degradation_15deg': deg0 - pose_results['rot_15deg']['mean_auroc'],
    }
    out = RESULTS_DIR / 'round45_pose_robustness.json'
    out.write_text(json.dumps(result, indent=2))
    print(f" Saved → {out}  ({elapsed/60:.1f} min)")
    return result


# ─────────────────────────────────────────────────────
# Experiment 5 — Scale Ablation (4 scales vs 5 scales)
# ─────────────────────────────────────────────────────

def exp5_scale_ablation():
    """
    Compare [4,8,16,32] (4 scales) vs [4,8,16,32,64] (5 scales) on both datasets.
    Also test GLARE-v2: 15D per scale at 5 scales = 75D total.
    """
    print("\n" + "="*65)
    print("EXP 5: Scale Ablation — 4-scale vs 5-scale GLARE")
    print("="*65)
    base_cfg = dict(n_shots=8, n_seeds=5, n_points=2048)
    t0 = time.time()

    r_4s_r3d = run_dataset_cached(REAL3D_ROOT,  'real3d',  scales=[4, 8, 16, 32],
                                  label='4scale-R3D',  **base_cfg)
    r_5s_r3d = run_dataset_cached(REAL3D_ROOT,  'real3d',  scales=[4, 8, 16, 32, 64],
                                  label='5scale-R3D',  **base_cfg)
    r_4s_mvt = run_dataset_cached(MVTEC3D_ROOT, 'mvtec3d', scales=[4, 8, 16, 32],
                                  label='4scale-MVT',  **base_cfg)
    r_5s_mvt = run_dataset_cached(MVTEC3D_ROOT, 'mvtec3d', scales=[4, 8, 16, 32, 64],
                                  label='5scale-MVT',  **base_cfg)

    elapsed = time.time() - t0
    g_r3d = r_5s_r3d['mean_auroc'] - r_4s_r3d['mean_auroc']
    g_mvt = r_5s_mvt['mean_auroc'] - r_4s_mvt['mean_auroc']
    print(f"\n  5-scale gain  Real3D: {g_r3d:+.4f}   MVTec: {g_mvt:+.4f}")

    result = {
        'experiment': 'scale_ablation', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device), 'config': base_cfg, 'elapsed_s': elapsed,
        '4scale': {'Real3D-AD': r_4s_r3d, 'MVTec3D-AD': r_4s_mvt},
        '5scale': {'Real3D-AD': r_5s_r3d, 'MVTec3D-AD': r_5s_mvt},
        'summary': {
            'r3d_4s': r_4s_r3d['mean_auroc'], 'r3d_5s': r_5s_r3d['mean_auroc'],
            'gain_r3d': g_r3d,
            'mvt_4s': r_4s_mvt['mean_auroc'], 'mvt_5s': r_5s_mvt['mean_auroc'],
            'gain_mvt': g_mvt,
        }
    }
    out = RESULTS_DIR / 'round45_scale_ablation.json'
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {out}  ({elapsed/60:.1f} min)")
    return result


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Round 45 GPU Experiments')
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiments: all, 1, 2, 3, 4, 5, or comma-separated')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  ROUND 45 — GPU-Optimized Comprehensive Experiments")
    print(f"  Device: {device}  |  {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"  Running experiments: {args.exp}")
    print("="*70)

    t_total = time.time()
    all_results = {}

    exps_to_run = (
        {1, 2, 3, 4, 5} if args.exp == 'all'
        else {int(e.strip()) for e in args.exp.split(',')}
    )

    if 1 in exps_to_run:
        all_results['exp1_multiseed']     = exp1_multiseed()
    if 2 in exps_to_run:
        all_results['exp2_dams_ablation'] = exp2_dams_ablation()
    if 3 in exps_to_run:
        all_results['exp3_shot']          = exp3_shot_sensitivity()
    if 4 in exps_to_run:
        all_results['exp4_pose']          = exp4_pose_robustness()
    if 5 in exps_to_run:
        all_results['exp5_scale']         = exp5_scale_ablation()

    total_elapsed = time.time() - t_total

    # ── Summary ──
    print("\n" + "="*70)
    print("  ROUND 45 SUMMARY")
    print("="*70)

    if 'exp1_multiseed' in all_results:
        s = all_results['exp1_multiseed']['summary']
        print(f"  [Exp1] 10-seed  Real3D: {s['real3d_mean']:.4f}±{s['real3d_std']:.4f}  "
              f"MVTec: {s['mvtec_mean']:.4f}±{s['mvtec_std']:.4f}")

    if 'exp2_dams_ablation' in all_results:
        s = all_results['exp2_dams_ablation']['summary']
        print(f"  [Exp2] DAMS gain  Real3D: {s['gain_r3d']:+.4f}  MVTec: {s['gain_mvt']:+.4f}")

    if 'exp3_shot' in all_results:
        s = all_results['exp3_shot']['summary']
        shot_str = '  '.join([f"{k}:{v:.4f}" for k, v in s.items()])
        print(f"  [Exp3] Shot sensitivity: {shot_str}")

    if 'exp4_pose' in all_results:
        r = all_results['exp4_pose']
        print(f"  [Exp4] Pose degradation  ±5°: {r['degradation_5deg']:+.4f}  "
              f"±15°: {r['degradation_15deg']:+.4f}")

    if 'exp5_scale' in all_results:
        s = all_results['exp5_scale']['summary']
        print(f"  [Exp5] Scale  R3D 4s→5s: {s['gain_r3d']:+.4f}  "
              f"MVT 4s→5s: {s['gain_mvt']:+.4f}")

    print(f"\n  Total time: {total_elapsed/60:.1f} min")

    # Save combined summary
    summary_path = RESULTS_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'round': 45,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'total_elapsed_s': total_elapsed,
            'experiments': list(all_results.keys()),
            'summaries': {k: v.get('summary', {}) for k, v in all_results.items()}
        }, f, indent=2)
    print(f" Combined summary → {summary_path}")


if __name__ == '__main__':
    main()
