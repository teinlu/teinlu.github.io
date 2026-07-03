#!/usr/bin/env python3
"""
Round 33: GLARE GPU-Accelerated Benchmark
Uses PyTorch CUDA for fast point cloud processing.

Author: ARIS Research Pipeline
Date: 2026-05-19
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch import Tensor

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import tifffile
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_tiff_to_tensor(file_path, n_points=2048, device='cuda'):
    """Load TIFF and convert to GPU tensor."""
    try:
        data = tifffile.imread(str(file_path))
        
        if data.ndim == 3 and data.shape[2] == 3:
            points = data.reshape(-1, 3)
            valid = ~np.any(np.isnan(points), axis=1) & ~np.all(points == 0, axis=1)
            points = points[valid].astype(np.float32)
        elif data.ndim == 2:
            h, w = data.shape
            valid = np.isfinite(data) & (data > 0)
            y, x = np.where(valid)
            points = np.stack([x, y, data[valid]], axis=1).astype(np.float32)
        else:
            return None
        
        if len(points) < 100:
            return None
        
        # Normalize
        center = points.mean(axis=0)
        points = points - center
        scale = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if scale > 0:
            points = points / scale
        
        # Subsample
        if len(points) > n_points:
            idx = np.random.choice(len(points), n_points, replace=False)
            points = points[idx]
        
        return torch.from_numpy(points).to(device)
    except Exception as e:
        return None


def knn_gpu(points: Tensor, k: int) -> tuple:
    """GPU-accelerated k-NN using PyTorch."""
    # points: (N, 3)
    # Compute pairwise distances
    dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)  # (N, N)
    
    # Get k+1 nearest (including self)
    distances, indices = torch.topk(dists, k + 1, largest=False, dim=1)
    
    return distances, indices


def extract_geometric_features_gpu(points: Tensor, k: int = 8) -> Tensor:
    """Extract geometric features using GPU."""
    n_points = points.shape[0]
    k_actual = min(k + 1, n_points)
    
    # k-NN on GPU
    distances, indices = knn_gpu(points, k_actual - 1)
    
    # Remove self (first neighbor)
    neighbor_dists = distances[:, 1:]  # (N, k)
    neighbor_idx = indices[:, 1:]  # (N, k)
    
    # Get neighbor points
    neighbors = points[neighbor_idx]  # (N, k, 3)
    
    # Centered neighbors
    centered = neighbors - points.unsqueeze(1)  # (N, k, 3)
    
    # Compute covariance matrices
    # cov = centered.T @ centered / k
    cov = torch.bmm(centered.transpose(1, 2), centered) / (k_actual - 1)  # (N, 3, 3)
    
    # Eigenvalue decomposition (batch)
    try:
        eigenvalues = torch.linalg.eigvalsh(cov)  # (N, 3)
        eigenvalues = torch.sort(eigenvalues, dim=1, descending=True).values
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
    except:
        eigenvalues = torch.ones(n_points, 3, device=points.device) / 3
    
    l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    total = l1 + l2 + l3 + 1e-10
    
    # Geometric features
    linearity = (l1 - l2) / (l1 + 1e-8)
    planarity = (l2 - l3) / (l1 + 1e-8)
    sphericity = l3 / (l1 + 1e-8)
    anisotropy = (l1 - l3) / (l1 + 1e-8)
    omnivariance = torch.pow(l1 * l2 * l3, 1/3)
    eigenentropy = -torch.sum(eigenvalues/total.unsqueeze(1) * torch.log(eigenvalues/total.unsqueeze(1) + 1e-10), dim=1)
    curvature = l3 / total
    
    # Distance features
    dist_mean = neighbor_dists.mean(dim=1)
    dist_std = neighbor_dists.std(dim=1)
    dist_max = neighbor_dists.max(dim=1).values
    dist_min = neighbor_dists.min(dim=1).values
    
    # Height features
    z_std = neighbors[:, :, 2].std(dim=1)
    z_mean = neighbors[:, :, 2].mean(dim=1)
    
    # Normalized eigenvalues
    norm_l1 = l1 / total
    norm_l2 = l2 / total
    norm_l3 = l3 / total
    
    # Stack features
    features = torch.stack([
        linearity, planarity, sphericity, anisotropy,
        omnivariance, eigenentropy, curvature,
        dist_mean, dist_std, dist_max, dist_min,
        z_std, z_mean,
        norm_l1, norm_l2, norm_l3,
        points[:, 2]  # z position
    ], dim=1)
    
    return features


def extract_multiscale_features_gpu(points: Tensor, scales=[4, 8, 16, 32]) -> Tensor:
    """Extract multi-scale features on GPU."""
    all_features = []
    for k in scales:
        features = extract_geometric_features_gpu(points, k)
        all_features.append(features)
    return torch.cat(all_features, dim=1)


class GLAREGpu:
    """GLARE with GPU acceleration."""
    
    def __init__(self, n_points=2048, scales=[4, 8, 16, 32], memory_size=5000, k_nn=5, device='cuda'):
        self.n_points = n_points
        self.scales = scales
        self.memory_size = memory_size
        self.k_nn = k_nn
        self.device = device
        self.memory_bank = None
        self.scaler = StandardScaler()
    
    def fit(self, point_clouds):
        """Fit on normal samples."""
        all_features = []
        
        for pc in point_clouds:
            if isinstance(pc, np.ndarray):
                pc = torch.from_numpy(pc).float().to(self.device)
            features = extract_multiscale_features_gpu(pc, self.scales)
            all_features.append(features.cpu().numpy())
        
        all_features = np.vstack(all_features)
        all_features = np.nan_to_num(all_features, nan=0)
        all_features = self.scaler.fit_transform(all_features)
        
        # Subsample for memory bank
        if len(all_features) > self.memory_size:
            idx = np.random.choice(len(all_features), self.memory_size, replace=False)
            self.memory_bank = torch.from_numpy(all_features[idx]).float().to(self.device)
        else:
            self.memory_bank = torch.from_numpy(all_features).float().to(self.device)
        
        return self
    
    def score(self, point_cloud):
        """Compute anomaly score using GPU."""
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float().to(self.device)
        
        features = extract_multiscale_features_gpu(point_cloud, self.scales)
        features_np = features.cpu().numpy()
        features_np = np.nan_to_num(features_np, nan=0)
        features_np = self.scaler.transform(features_np)
        features = torch.from_numpy(features_np).float().to(self.device)
        
        # k-NN scoring on GPU
        dists = torch.cdist(features, self.memory_bank)  # (N, M)
        k = min(self.k_nn, self.memory_bank.shape[0])
        nearest_k = torch.topk(dists, k, largest=False, dim=1).values  # (N, k)
        point_scores = nearest_k.mean(dim=1)  # (N,)
        
        # Top-95 percentile aggregation
        point_scores_np = point_scores.cpu().numpy()
        threshold = np.percentile(point_scores_np, 95)
        top_scores = point_scores_np[point_scores_np >= threshold]
        
        return np.mean(top_scores) if len(top_scores) > 0 else np.mean(point_scores_np)


def evaluate_category_gpu(cat_name, data_dir, n_shots=8, n_seeds=3, device='cuda'):
    """Evaluate a single category using GPU."""
    cat_path = Path(data_dir) / cat_name
    
    # Find train files
    train_dir = cat_path / 'train' / 'good' / 'xyz'
    if not train_dir.exists():
        return cat_name, []
    
    train_files = sorted(train_dir.glob('*.tiff'))
    
    # Find test files
    test_dir = cat_path / 'test'
    if not test_dir.exists():
        return cat_name, []
    
    test_files, test_labels = [], []
    for subdir in sorted(test_dir.iterdir()):
        if subdir.is_dir():
            xyz_dir = subdir / 'xyz'
            if xyz_dir.exists():
                label = 0 if subdir.name == 'good' else 1
                for f in sorted(xyz_dir.glob('*.tiff')):
                    test_files.append(f)
                    test_labels.append(label)
    
    if len(train_files) < n_shots or len(test_files) < 2:
        return cat_name, []
    
    aurocs = []
    
    for seed in range(n_seeds):
        np.random.seed(seed + 42)
        torch.manual_seed(seed + 42)
        
        # Load train samples
        train_idx = np.random.choice(len(train_files), min(n_shots, len(train_files)), replace=False)
        train_pcs = []
        for idx in train_idx:
            pc = load_tiff_to_tensor(train_files[idx], device=device)
            if pc is not None:
                train_pcs.append(pc)
        
        if len(train_pcs) < 1:
            continue
        
        # Fit model
        model = GLAREGpu(n_points=2048, scales=[4, 8, 16, 32], device=device)
        model.fit(train_pcs)
        
        # Evaluate
        scores, labels = [], []
        for f, label in zip(test_files, test_labels):
            pc = load_tiff_to_tensor(f, device=device)
            if pc is not None:
                score = model.score(pc)
                scores.append(score)
                labels.append(label)
        
        if len(scores) > 0 and len(set(labels)) > 1:
            auroc = roc_auc_score(labels, scores)
            aurocs.append(auroc)
            print(f"  {cat_name} seed={seed}: AUROC={auroc:.4f}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return cat_name, aurocs


def main():
    """Main benchmark."""
    print("="*70)
    print("GLARE GPU-Accelerated Benchmark - Round 33")
    print("="*70)
    
    # Data paths
    real3d_dir = Path('/home/cxs/桌面/aris2/Real3D-mvtec')
    mvtec3d_dir = Path('/home/cxs/桌面/aris2/MVTec3d')
    results_dir = Path('/home/cxs/桌面/aris2/results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'GLARE-GPU',
        'device': str(device),
        'config': {
            'n_points': 2048,
            'scales': [4, 8, 16, 32],
            'n_shots': 8,
            'n_seeds': 3
        },
        'datasets': {}
    }
    
    # Categories
    real3d_cats = ['airplane', 'candybar', 'car', 'chicken', 'diamond', 'duck',
                   'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees']
    
    mvtec3d_cats = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'foam', 'peach', 'potato', 'rope', 'tire']
    
    print("\n" + "="*50)
    print("Evaluating Real3D-AD")
    print("="*50)
    
    real3d_results = {}
    all_aurocs = []
    
    start_time = time.time()
    
    for cat in real3d_cats:
        cat_name, aurocs = evaluate_category_gpu(cat, real3d_dir, device=device)
        if aurocs:
            mean_auroc = np.mean(aurocs)
            std_auroc = np.std(aurocs)
            real3d_results[cat] = {
                'mean': float(mean_auroc),
                'std': float(std_auroc),
                'aurocs': [float(a) for a in aurocs]
            }
            all_aurocs.extend(aurocs)
            print(f"  {cat}: {mean_auroc:.4f} ± {std_auroc:.4f}")
    
    if all_aurocs:
        real3d_mean = np.mean(all_aurocs)
        real3d_std = np.std(all_aurocs)
        results['datasets']['Real3D-AD'] = {
            'categories': real3d_results,
            'mean_auroc': float(real3d_mean),
            'std_auroc': float(real3d_std)
        }
        print(f"\nReal3D-AD Mean: {real3d_mean:.4f} ± {real3d_std:.4f}")
    
    print("\n" + "="*50)
    print("Evaluating MVTec3D-AD")
    print("="*50)
    
    mvtec3d_results = {}
    all_aurocs = []
    
    for cat in mvtec3d_cats:
        cat_name, aurocs = evaluate_category_gpu(cat, mvtec3d_dir, device=device)
        if aurocs:
            mean_auroc = np.mean(aurocs)
            std_auroc = np.std(aurocs)
            mvtec3d_results[cat] = {
                'mean': float(mean_auroc),
                'std': float(std_auroc),
                'aurocs': [float(a) for a in aurocs]
            }
            all_aurocs.extend(aurocs)
            print(f"  {cat}: {mean_auroc:.4f} ± {std_auroc:.4f}")
    
    if all_aurocs:
        mvtec3d_mean = np.mean(all_aurocs)
        mvtec3d_std = np.std(all_aurocs)
        results['datasets']['MVTec3D-AD'] = {
            'categories': mvtec3d_results,
            'mean_auroc': float(mvtec3d_mean),
            'std_auroc': float(mvtec3d_std)
        }
        print(f"\nMVTec3D-AD Mean: {mvtec3d_mean:.4f} ± {mvtec3d_std:.4f}")
    
    elapsed = time.time() - start_time
    results['elapsed_seconds'] = elapsed
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if 'Real3D-AD' in results['datasets']:
        print(f"Real3D-AD:  {results['datasets']['Real3D-AD']['mean_auroc']:.4f}")
    if 'MVTec3D-AD' in results['datasets']:
        print(f"MVTec3D-AD: {results['datasets']['MVTec3D-AD']['mean_auroc']:.4f}")
    print(f"Time: {elapsed/60:.1f} minutes")
    
    # SOTA comparison
    print("\n" + "-"*50)
    print("SOTA Comparison (Real3D-AD):")
    print("-"*50)
    sota_methods = {
        'Real3D baseline (RGB-D)': 0.704,
        'PatchCore+FPFH+raw': 0.682,
        'BTF_FPFH': 0.635,
        'PointCore': 0.625,
    }
    
    our_auroc = results['datasets'].get('Real3D-AD', {}).get('mean_auroc', 0)
    for method, auroc in sota_methods.items():
        delta = our_auroc - auroc
        status = "✓" if delta > 0 else "✗"
        print(f"  {method}: {auroc:.3f} (Δ={delta:+.3f}) {status}")
    print(f"  GLARE-GPU (Ours): {our_auroc:.4f}")
    
    # Save results
    output_file = results_dir / 'round33_glare_gpu.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
