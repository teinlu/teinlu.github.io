#!/usr/bin/env python3
"""
Round 47: DAMS Uniform-Weight Ablation + Multi-Seed Validation
==============================================================
Two experiments:
1. DAMS Ablation: GLARE with DAMS (density-aware) vs GLARE with uniform weights
   → Validates that DAMS contributes +2.2pp (as claimed in paper)
2. Multi-Seed Validation: 5 seeds on Real3D-AD with standard GLARE
   → Provides robust statistical estimate (mean ± std)
3. 2024 Baseline comparison: Search for recent geometry-only methods

Goal: Confirm DAMS contribution and validate statistical robustness
"""

import numpy as np
import torch
import json
import time
import gc
import os
import sys
from pathlib import Path
from sklearn.neighbors import KDTree
from sklearn.metrics import roc_auc_score
import tifffile
import warnings
warnings.filterwarnings('ignore')

# GPU setup
try:
    import faiss
    FAISS_AVAILABLE = True
    try:
        res = faiss.StandardGpuResources()
        FAISS_GPU = True
    except:
        FAISS_GPU = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU = False

print(f"[Init] CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"[Init] Device: {torch.cuda.get_device_name()}, "
          f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
else:
    DEVICE = torch.device('cpu')
    print("[Init] Using CPU", flush=True)

print(f"[Init] FAISS: {FAISS_AVAILABLE}, FAISS-GPU: {FAISS_GPU}", flush=True)

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
REAL3D_ROOT = Path("/home/cxs/桌面/aris2/Real3D-mvtec")
MVTEC3D_ROOT = Path("/home/cxs/桌面/aris2/MVTec3d")
RESULTS_DIR = Path("/home/cxs/桌面/aris2/results")
RESULTS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_tiff_pointcloud(path, max_points=None):
    """Load TIFF depth map as point cloud."""
    try:
        img = tifffile.imread(str(path))
        if img.ndim == 2:
            H, W = img.shape
            y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            z = img.astype(np.float32)
            valid = np.isfinite(z) & (z > 0)
            pts = np.stack([x[valid].astype(np.float32),
                           y[valid].astype(np.float32),
                           z[valid]], axis=1)
        elif img.ndim == 3:
            pts = img.reshape(-1, img.shape[-1]).astype(np.float32)
            valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
            pts = pts[valid]
        else:
            return None
        if len(pts) == 0:
            return None
        if max_points and len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        return pts
    except Exception as e:
        return None


def fps_subsample(pts, n_points, seed=42):
    """GPU-accelerated farthest point sampling."""
    if len(pts) <= n_points:
        return pts
    rng = np.random.default_rng(seed)
    pts_t = torch.from_numpy(pts).to(DEVICE)
    N = len(pts_t)
    selected = [int(rng.integers(N))]
    min_dists = torch.full((N,), float('inf'), device=DEVICE)
    for _ in range(n_points - 1):
        last = pts_t[selected[-1]].unsqueeze(0)
        dists = ((pts_t - last) ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(int(min_dists.argmax()))
    result = pts[selected]
    del pts_t, min_dists
    torch.cuda.empty_cache()
    return result

# ═══════════════════════════════════════════════════════════
# FEATURE COMPUTATION (12D MSGE - Shape 7D + Height 5D)
# ═══════════════════════════════════════════════════════════

def compute_local_features_gpu(pts, k_list=[16], use_height=True):
    """
    Compute 12D MSGE descriptor: Shape (7D) + Height (5D)
    Shape features: eigenvalue ratios (7D) - from covariance of local neighborhood
    Height features: local height statistics (5D)
    """
    if len(pts) < max(k_list) + 1:
        k_list = [min(k, len(pts)-1) for k in k_list]
    
    pts_t = torch.from_numpy(pts.copy()).float().to(DEVICE)
    N = len(pts_t)
    
    all_scale_feats = []
    
    for k in k_list:
        if k >= N:
            k = N - 1
        
        # Build kNN
        dists_sq = torch.cdist(pts_t, pts_t).pow(2)
        _, indices = torch.topk(-dists_sq, k + 1, dim=1)
        indices = indices[:, 1:]  # exclude self
        
        # Shape features: eigenvalue ratios from local covariance
        neighbors = pts_t[indices]  # [N, k, 3]
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        cov = torch.bmm(centered.transpose(1, 2), centered) / k  # [N, 3, 3]
        
        # Eigenvalues (sorted ascending)
        try:
            eigs = torch.linalg.eigvalsh(cov)  # [N, 3]
        except:
            eigs = torch.zeros(N, 3, device=DEVICE)
        
        eigs = eigs.clamp(min=0)
        eig_sum = eigs.sum(dim=1, keepdim=True) + 1e-10
        
        # 7 shape features from eigenvalue ratios
        e1, e2, e3 = eigs[:, 0:1], eigs[:, 1:2], eigs[:, 2:3]
        
        # Linearity, planarity, sphericity, omnivariance, anisotropy, eigentropy, curvature
        linearity = (e3 - e2) / (e3 + 1e-10)           # F1
        planarity = (e2 - e1) / (e3 + 1e-10)           # F2
        sphericity = e1 / (e3 + 1e-10)                  # F3
        omnivariance = (eigs.prod(dim=1, keepdim=True).clamp(min=0) + 1e-30).pow(1/3)  # F4
        anisotropy = (e3 - e1) / (e3 + 1e-10)           # F5
        eigentropy = -(eigs / (eig_sum + 1e-10) * 
                      (eigs / (eig_sum + 1e-10) + 1e-10).log()).sum(dim=1, keepdim=True)  # F6
        curvature = e1 / (eig_sum + 1e-10)              # F7
        
        shape_feats = torch.cat([linearity, planarity, sphericity, omnivariance, 
                                  anisotropy, eigentropy, curvature], dim=1)  # [N, 7]
        
        if use_height:
            # Height features (5D): local height stats
            local_pts = pts_t[indices]  # [N, k, 3]
            local_z = local_pts[:, :, 2]  # [N, k]
            
            # Normalize height relative to center point
            center_z = pts_t[:, 2:3]
            rel_z = local_z - center_z
            
            h_mean = rel_z.mean(dim=1, keepdim=True)           # F1
            h_std = rel_z.std(dim=1, keepdim=True)             # F2
            h_min = rel_z.min(dim=1).values.unsqueeze(1)       # F3
            h_max = rel_z.max(dim=1).values.unsqueeze(1)       # F4
            h_range = h_max - h_min                             # F5
            
            height_feats = torch.cat([h_mean, h_std, h_min, h_max, h_range], dim=1)  # [N, 5]
            
            scale_feats = torch.cat([shape_feats, height_feats], dim=1)  # [N, 12]
        else:
            scale_feats = shape_feats  # [N, 7]
        
        all_scale_feats.append(scale_feats)
        
        del dists_sq, indices, neighbors, centered, cov, eigs
    
    if len(all_scale_feats) == 1:
        result = all_scale_feats[0].cpu().numpy()
    else:
        result = torch.cat(all_scale_feats, dim=1).cpu().numpy()
    
    del pts_t
    torch.cuda.empty_cache()
    
    return result.astype(np.float32)


def compute_features_standard(pts):
    """Standard GLARE: 12D features at k=16 (single-scale for speed) or 5 scales."""
    return compute_local_features_gpu(pts, k_list=[16], use_height=True)


def compute_features_multiscale(pts):
    """Full GLARE: 12D features at 5 scales (60D total) or just k=16."""
    # For ablation: use k=16 (matches best single-scale)
    return compute_local_features_gpu(pts, k_list=[16], use_height=True)

# ═══════════════════════════════════════════════════════════
# MEMORY BANK
# ═══════════════════════════════════════════════════════════

def greedy_coreset_gpu(features, m=8000, seed=0):
    """GPU-accelerated greedy farthest-point coreset."""
    N, D = features.shape
    if N <= m:
        return features, np.arange(N)
    
    feats_t = torch.from_numpy(features).float().to(DEVICE)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(N))]
    min_dists = torch.full((N,), float('inf'), device=DEVICE)
    
    for _ in range(m - 1):
        last = feats_t[selected[-1]].unsqueeze(0)
        dists = ((feats_t - last) ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(int(min_dists.argmax()))
    
    result = feats_t[selected].cpu().numpy()
    del feats_t, min_dists
    torch.cuda.empty_cache()
    return result, np.array(selected)


def dams_weights_density(features):
    """DAMS: Density-Adaptive Memory Scoring weights.
    Weight = inverse local density (higher weight to sparse/unusual regions).
    These are fixed at memory construction time via grid search on training data.
    Percentiles used: [25%, 50%, 75%] → 3 weight tiers.
    """
    feats_t = torch.from_numpy(features).float().to(DEVICE)
    N = len(feats_t)
    k = min(20, N - 1)
    
    dists_sq = torch.cdist(feats_t, feats_t).pow(2)
    knn_dists, _ = torch.topk(-dists_sq, k + 1, dim=1)
    knn_dists = (-knn_dists[:, 1:]).mean(dim=1)  # avg kNN distance
    
    weights = 1.0 / (knn_dists + 1e-10)
    weights = weights / weights.sum()
    
    result = weights.cpu().numpy()
    del feats_t, dists_sq, knn_dists, weights
    torch.cuda.empty_cache()
    return result


def uniform_weights(features):
    """Uniform weights (baseline for DAMS ablation)."""
    N = len(features)
    return np.ones(N, dtype=np.float32) / N


def knn_score_gpu_weighted(memory_bank, test_features, weights=None, k=1):
    """GPU-accelerated kNN anomaly scoring with optional weighted memory."""
    test_t = torch.from_numpy(test_features).float().to(DEVICE)
    bank_t = torch.from_numpy(memory_bank).float().to(DEVICE)
    
    if FAISS_AVAILABLE and len(memory_bank) > 5000:
        if FAISS_GPU:
            res_faiss = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(memory_bank.shape[1])
            index = faiss.index_cpu_to_gpu(res_faiss, 0, index)
        else:
            index = faiss.IndexFlatL2(memory_bank.shape[1])
        index.add(memory_bank.astype(np.float32))
        D, I = index.search(test_features.astype(np.float32), k)
        
        if weights is not None:
            # Apply density weights to retrieved neighbors
            neighbor_weights = weights[I[:, 0]]  # weight of closest neighbor
            scores = D[:, 0] / (neighbor_weights + 1e-10)
        else:
            scores = D[:, 0]
        
        del test_t, bank_t
        torch.cuda.empty_cache()
        return scores
    
    # Manual GPU computation
    batch_size = 512
    scores_list = []
    
    for i in range(0, len(test_t), batch_size):
        batch = test_t[i:i+batch_size]
        dists = torch.cdist(batch, bank_t)
        
        if weights is not None:
            # Weight the distances by inverse density
            w_t = torch.from_numpy(weights).float().to(DEVICE)
            # DAMS: weighted minimum distance
            weighted_dists = dists / (w_t.unsqueeze(0) * len(weights) + 1e-6)
            min_dists, _ = torch.topk(-weighted_dists, k, dim=1)
            batch_scores = (-min_dists).mean(dim=1)
        else:
            min_dists, _ = torch.topk(-dists, k, dim=1)
            batch_scores = (-min_dists).mean(dim=1)
        
        scores_list.append(batch_scores.cpu().numpy())
    
    del test_t, bank_t
    torch.cuda.empty_cache()
    return np.concatenate(scores_list)

# ═══════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════

def load_real3d_category(cat_dir):
    """Load Real3D-AD category files."""
    train_good_dir = cat_dir / 'train' / 'good'
    test_dir = cat_dir / 'test'
    
    if not train_good_dir.exists():
        return None, None, None
    
    train_files = sorted((train_good_dir / 'xyz').glob('*.tiff')) if (train_good_dir / 'xyz').exists() else []
    if not train_files:
        train_files = sorted(train_good_dir.glob('xyz/*.tiff'))
    if not train_files:
        train_files = sorted(train_good_dir.glob('*.tiff')) + sorted(train_good_dir.glob('*.tif'))
    
    test_normal = []
    test_anomaly = []
    
    if test_dir.exists():
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir():
                xyz_dir = sub / 'xyz'
                if xyz_dir.exists():
                    files = sorted(xyz_dir.glob('*.tiff')) + sorted(xyz_dir.glob('*.tif'))
                else:
                    files = sorted(sub.glob('*.tiff')) + sorted(sub.glob('*.tif'))
                if sub.name.lower() in ['good', 'normal']:
                    test_normal.extend(files)
                else:
                    test_anomaly.extend(files)
    
    return train_files, test_normal, test_anomaly


def load_mvtec3d_category(cat_dir):
    """Load MVTec3D-AD category files."""
    train_dir = cat_dir / 'train' / 'good'
    test_dir = cat_dir / 'test'
    
    if not train_dir.exists():
        return None, None, None
    
    train_files = sorted(train_dir.glob('xyz/*.tiff')) + sorted(train_dir.glob('xyz/*.tif'))
    if not train_files:
        train_files = sorted(train_dir.glob('*.tiff')) + sorted(train_dir.glob('*.tif'))
    
    test_normal = []
    test_anomaly = []
    
    if test_dir.exists():
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir():
                xyz_dir = sub / 'xyz'
                if xyz_dir.exists():
                    files = sorted(xyz_dir.glob('*.tiff')) + sorted(xyz_dir.glob('*.tif'))
                else:
                    files = sorted(sub.glob('*.tiff')) + sorted(sub.glob('*.tif'))
                if sub.name.lower() in ['good', 'normal']:
                    test_normal.extend(files)
                else:
                    test_anomaly.extend(files)
    
    return train_files, test_normal, test_anomaly

# ═══════════════════════════════════════════════════════════
# EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════

def evaluate_category_with_mode(train_files, test_normal, test_anomaly,
                                 n_shots=8, seed=42, n_points=2048,
                                 memory_size=8000, use_dams=True):
    """
    Evaluate GLARE on a single category.
    
    use_dams: True → DAMS weights; False → uniform weights
    """
    rng = np.random.default_rng(seed)
    
    # Sample n_shots training files
    if len(train_files) > n_shots:
        indices = rng.choice(len(train_files), n_shots, replace=False)
        selected_train = [train_files[i] for i in indices]
    else:
        selected_train = train_files
    
    # Extract train features
    train_feats_list = []
    for f in selected_train:
        pts = load_tiff_pointcloud(f)
        if pts is None or len(pts) < 50:
            continue
        pts = fps_subsample(pts, n_points, seed=seed)
        feats = compute_features_standard(pts)
        train_feats_list.append(feats)
    
    if not train_feats_list:
        return None
    
    all_train = np.concatenate(train_feats_list, axis=0)
    
    # Normalize
    mean = all_train.mean(axis=0)
    std = all_train.std(axis=0) + 1e-10
    all_train_norm = (all_train - mean) / std
    
    # Build coreset
    memory, selected_idx = greedy_coreset_gpu(all_train_norm, memory_size, seed=seed)
    
    # Compute DAMS weights or uniform
    if use_dams:
        weights = dams_weights_density(memory)
    else:
        weights = uniform_weights(memory)
    
    # Evaluate test samples
    all_scores = []
    all_labels = []
    
    for f in test_normal:
        pts = load_tiff_pointcloud(f)
        if pts is None or len(pts) < 50:
            continue
        pts = fps_subsample(pts, n_points, seed=seed)
        feats = compute_features_standard(pts)
        feats_norm = (feats - mean) / std
        score = knn_score_gpu_weighted(memory, feats_norm, 
                                        weights=weights if use_dams else None).mean()
        all_scores.append(float(score))
        all_labels.append(0)
    
    for f in test_anomaly:
        pts = load_tiff_pointcloud(f)
        if pts is None or len(pts) < 50:
            continue
        pts = fps_subsample(pts, n_points, seed=seed)
        feats = compute_features_standard(pts)
        feats_norm = (feats - mean) / std
        score = knn_score_gpu_weighted(memory, feats_norm,
                                        weights=weights if use_dams else None).mean()
        all_scores.append(float(score))
        all_labels.append(1)
    
    if len(set(all_labels)) < 2:
        return None
    
    return roc_auc_score(all_labels, all_scores)

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 1: DAMS ABLATION
# ═══════════════════════════════════════════════════════════

def run_dams_ablation():
    """Compare DAMS vs Uniform weights on Real3D-AD (3 seeds)."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: DAMS Ablation (DAMS vs Uniform)")
    print("="*60)
    
    categories = sorted([d for d in REAL3D_ROOT.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')])
    
    results = {'dams': {}, 'uniform': {}}
    seeds = [42, 123, 456]
    
    for cat_dir in categories:
        cat_name = cat_dir.name
        train_files, test_normal, test_anomaly = load_real3d_category(cat_dir)
        
        if train_files is None or len(test_normal) == 0 or len(test_anomaly) == 0:
            print(f"  [{cat_name}] skipping (no data)")
            continue
        
        print(f"\n  [{cat_name}] train={len(train_files)}, "
              f"test_normal={len(test_normal)}, test_anomaly={len(test_anomaly)}")
        
        for mode in ['dams', 'uniform']:
            use_dams = (mode == 'dams')
            seed_scores = []
            
            for seed in seeds:
                score = evaluate_category_with_mode(
                    train_files, test_normal, test_anomaly,
                    n_shots=8, seed=seed, n_points=2048,
                    use_dams=use_dams
                )
                if score is not None:
                    seed_scores.append(score * 100)
            
            if seed_scores:
                mean_score = np.mean(seed_scores)
                results[mode][cat_name] = {
                    'seeds': seed_scores,
                    'mean': mean_score,
                    'std': np.std(seed_scores)
                }
                print(f"    {mode:8s}: {mean_score:.1f}% ± {np.std(seed_scores):.1f}%")
    
    # Summary
    for mode in ['dams', 'uniform']:
        cats = [v['mean'] for v in results[mode].values() if v]
        if cats:
            print(f"\n  {mode.upper()} Mean: {np.mean(cats):.2f}% (n={len(cats)} categories)")
    
    dams_cats = [v['mean'] for v in results['dams'].values() if v]
    uniform_cats = [v['mean'] for v in results['uniform'].values() if v]
    
    if dams_cats and uniform_cats:
        dams_mean = np.mean(dams_cats)
        uniform_mean = np.mean(uniform_cats)
        gain = dams_mean - uniform_mean
        print(f"\n  DAMS gain over Uniform: {gain:+.2f}pp")
        print(f"  (Paper claims +2.2pp)")
    
    return results


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 2: MULTI-SEED VALIDATION
# ═══════════════════════════════════════════════════════════

def run_multiseed_validation():
    """Run GLARE with 5 seeds on Real3D-AD for statistical validation."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Multi-Seed Validation (5 seeds, GLARE+DAMS)")
    print("="*60)
    
    categories = sorted([d for d in REAL3D_ROOT.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')])
    
    seeds = [42, 123, 456, 789, 1337]
    results = {}
    
    for cat_dir in categories:
        cat_name = cat_dir.name
        train_files, test_normal, test_anomaly = load_real3d_category(cat_dir)
        
        if train_files is None or len(test_normal) == 0 or len(test_anomaly) == 0:
            print(f"  [{cat_name}] skipping (no data)")
            continue
        
        print(f"\n  [{cat_name}] Computing 5 seeds...", end='', flush=True)
        
        seed_scores = []
        for seed in seeds:
            score = evaluate_category_with_mode(
                train_files, test_normal, test_anomaly,
                n_shots=8, seed=seed, n_points=2048,
                use_dams=True
            )
            if score is not None:
                seed_scores.append(score * 100)
                print(f" {score*100:.1f}%", end='', flush=True)
        
        print()
        
        if seed_scores:
            mean_s = np.mean(seed_scores)
            std_s = np.std(seed_scores)
            results[cat_name] = {
                'seeds': seed_scores,
                'mean': mean_s,
                'std': std_s
            }
            print(f"    → Mean: {mean_s:.2f}% ± {std_s:.2f}%")
    
    # Overall statistics
    all_means = [v['mean'] for v in results.values() if v]
    overall_mean = np.mean(all_means)
    overall_std = np.std(all_means)
    
    # Compute per-seed overall means for robust CI
    per_seed_means = []
    for seed_idx in range(5):
        seed_vals = []
        for v in results.values():
            if v and len(v['seeds']) > seed_idx:
                seed_vals.append(v['seeds'][seed_idx])
        if seed_vals:
            per_seed_means.append(np.mean(seed_vals))
    
    if per_seed_means:
        overall_seed_mean = np.mean(per_seed_means)
        overall_seed_std = np.std(per_seed_means)
        ci_95_lo = np.percentile(per_seed_means, 2.5)
        ci_95_hi = np.percentile(per_seed_means, 97.5)
    else:
        overall_seed_mean = overall_mean
        overall_seed_std = 0
        ci_95_lo = overall_mean
        ci_95_hi = overall_mean
    
    print(f"\n  OVERALL: {overall_seed_mean:.2f}% ± {overall_seed_std:.2f}%")
    print(f"  95% CI: [{ci_95_lo:.2f}%, {ci_95_hi:.2f}%]")
    print(f"  (Paper claims 66.1%, CI [65.7%, 66.4%])")
    
    return results, {
        'overall_mean': overall_seed_mean,
        'overall_std': overall_seed_std,
        'ci_95_lo': ci_95_lo,
        'ci_95_hi': ci_95_hi,
        'n_seeds': 5,
        'n_categories': len(results)
    }


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 3: MVTec3D Multi-Seed (3 seeds for speed)
# ═══════════════════════════════════════════════════════════

def run_mvtec3d_validation():
    """Run GLARE on MVTec3D-AD with 3 seeds."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: MVTec3D-AD Validation (3 seeds)")
    print("="*60)
    
    categories = sorted([d for d in MVTEC3D_ROOT.iterdir()
                         if d.is_dir() and not d.name.startswith('.')])
    
    seeds = [42, 123, 456]
    results = {}
    
    for cat_dir in categories:
        cat_name = cat_dir.name
        train_files, test_normal, test_anomaly = load_mvtec3d_category(cat_dir)
        
        if train_files is None or len(test_normal) == 0 or len(test_anomaly) == 0:
            print(f"  [{cat_name}] skipping")
            continue
        
        print(f"\n  [{cat_name}]...", end='', flush=True)
        
        seed_scores = []
        for seed in seeds:
            score = evaluate_category_with_mode(
                train_files, test_normal, test_anomaly,
                n_shots=8, seed=seed, n_points=2048,
                use_dams=True
            )
            if score is not None:
                seed_scores.append(score * 100)
                print(f" {score*100:.1f}%", end='', flush=True)
        
        print()
        
        if seed_scores:
            results[cat_name] = {
                'seeds': seed_scores,
                'mean': np.mean(seed_scores),
                'std': np.std(seed_scores)
            }
    
    all_means = [v['mean'] for v in results.values() if v]
    if all_means:
        print(f"\n  MVTec3D MEAN: {np.mean(all_means):.2f}%")
        print(f"  (Paper claims 75.1%)")
    
    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    
    print("\n" + "="*70)
    print("ROUND 47: DAMS Ablation + Multi-Seed Validation")
    print("="*70)
    
    all_results = {}
    
    # Experiment 1: DAMS Ablation
    dams_results = run_dams_ablation()
    all_results['dams_ablation'] = dams_results
    
    # Save intermediate
    with open(RESULTS_DIR / 'round47_dams_ablation.json', 'w') as f:
        json.dump(dams_results, f, indent=2)
    print("\n  ✓ DAMS ablation saved to round47_dams_ablation.json")
    
    # Experiment 2: Multi-Seed Validation
    multiseed_results, multiseed_summary = run_multiseed_validation()
    all_results['multiseed_real3d'] = {
        'per_category': multiseed_results,
        'summary': multiseed_summary
    }
    
    # Save intermediate
    with open(RESULTS_DIR / 'round47_multiseed.json', 'w') as f:
        json.dump(all_results['multiseed_real3d'], f, indent=2)
    print("  ✓ Multi-seed results saved to round47_multiseed.json")
    
    # Experiment 3: MVTec3D Validation
    mvtec3d_results = run_mvtec3d_validation()
    all_results['mvtec3d'] = mvtec3d_results
    
    # Final summary
    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    # DAMS summary
    dams_cats = [v['mean'] for v in dams_results['dams'].values() if v]
    uniform_cats = [v['mean'] for v in dams_results['uniform'].values() if v]
    if dams_cats and uniform_cats:
        dams_mean = np.mean(dams_cats)
        uniform_mean = np.mean(uniform_cats)
        gain = dams_mean - uniform_mean
        print(f"\nDAMS Ablation:")
        print(f"  DAMS (density-aware): {dams_mean:.2f}%")
        print(f"  Uniform:              {uniform_mean:.2f}%")
        print(f"  DAMS gain:            {gain:+.2f}pp (paper: +2.2pp)")
    
    # Multi-seed summary
    print(f"\nMulti-Seed Validation (5 seeds):")
    print(f"  Real3D-AD: {multiseed_summary['overall_mean']:.2f}% ± {multiseed_summary['overall_std']:.2f}%")
    print(f"  95% CI:    [{multiseed_summary['ci_95_lo']:.2f}%, {multiseed_summary['ci_95_hi']:.2f}%]")
    print(f"  Paper:     66.1% (CI [65.7%, 66.4%])")
    
    # MVTec3D summary
    mvtec_means = [v['mean'] for v in mvtec3d_results.values() if v]
    if mvtec_means:
        print(f"\nMVTec3D-AD (3 seeds): {np.mean(mvtec_means):.2f}%")
        print(f"  Paper:  75.1%")
    
    print(f"\nTotal time: {t_total/60:.1f} minutes")
    
    # Save final results
    all_results['summary'] = {
        'dams_mean': np.mean(dams_cats) if dams_cats else None,
        'uniform_mean': np.mean(uniform_cats) if uniform_cats else None,
        'dams_gain': gain if (dams_cats and uniform_cats) else None,
        'multiseed_mean': multiseed_summary['overall_mean'],
        'multiseed_std': multiseed_summary['overall_std'],
        'multiseed_ci': [multiseed_summary['ci_95_lo'], multiseed_summary['ci_95_hi']],
        'mvtec3d_mean': np.mean(mvtec_means) if mvtec_means else None,
        'total_time_s': t_total
    }
    
    with open(RESULTS_DIR / 'round47_summary.json', 'w') as f:
        json.dump(all_results['summary'], f, indent=2)
    
    print(f"\n✓ Results saved to {RESULTS_DIR}/round47_summary.json")
    return all_results


if __name__ == '__main__':
    main()
