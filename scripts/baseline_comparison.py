#!/usr/bin/env python3
"""
Round 40: Baseline Comparison Study
====================================
Compare GLARE against proper baselines to establish true contribution.

Baselines:
1. Random baseline
2. PCA-based features
3. Local curvature only
4. GLARE (our method)

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
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: CUDA not available!")
    sys.exit(1)

import tifffile
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
        
        center = points.mean(axis=0)
        points = points - center
        scale = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if scale > 0:
            points = points / scale
        
        if len(points) > n_points:
            idx = np.random.choice(len(points), n_points, replace=False)
            points = points[idx]
        
        return torch.from_numpy(points).to(device)
    except:
        return None


def extract_geometric_features_gpu(points: Tensor, k: int = 8) -> Tensor:
    """Extract full geometric features - same as GLARE."""
    n_points = points.shape[0]
    k_actual = min(k + 1, n_points)
    
    dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
    distances, indices = torch.topk(dists, k_actual, largest=False, dim=1)
    
    neighbor_dists = distances[:, 1:]
    neighbor_idx = indices[:, 1:]
    neighbors = points[neighbor_idx]
    centered = neighbors - points.unsqueeze(1)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (k_actual - 1)
    
    try:
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = torch.sort(eigenvalues, dim=1, descending=True).values
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
    except:
        eigenvalues = torch.ones(n_points, 3, device=points.device) / 3
    
    l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    total = l1 + l2 + l3 + 1e-10
    
    linearity = (l1 - l2) / (l1 + 1e-8)
    planarity = (l2 - l3) / (l1 + 1e-8)
    sphericity = l3 / (l1 + 1e-8)
    anisotropy = (l1 - l3) / (l1 + 1e-8)
    omnivariance = torch.pow(l1 * l2 * l3, 1/3)
    eigenentropy = -torch.sum(eigenvalues/total.unsqueeze(1) * torch.log(eigenvalues/total.unsqueeze(1) + 1e-10), dim=1)
    curvature = l3 / total
    
    dist_mean = neighbor_dists.mean(dim=1)
    dist_std = neighbor_dists.std(dim=1)
    dist_max = neighbor_dists.max(dim=1).values
    dist_min = neighbor_dists.min(dim=1).values
    
    z_std = neighbors[:, :, 2].std(dim=1)
    z_mean = neighbors[:, :, 2].mean(dim=1)
    
    norm_l1 = l1 / total
    norm_l2 = l2 / total
    norm_l3 = l3 / total
    
    features = torch.stack([
        linearity, planarity, sphericity, anisotropy,
        omnivariance, eigenentropy, curvature,
        dist_mean, dist_std, dist_max, dist_min,
        z_std, z_mean,
        norm_l1, norm_l2, norm_l3,
        points[:, 2]
    ], dim=1)
    
    return features


class RandomBaseline:
    """Random scoring baseline."""
    def fit(self, point_clouds): return self
    def score(self, pc): return np.random.random()


class CurvatureBaseline:
    """Only curvature-based features (3D features: l1, l2, l3)."""
    
    def __init__(self, k=16, memory_size=5000, k_nn=5, device='cuda'):
        self.k = k
        self.memory_size = memory_size
        self.k_nn = k_nn
        self.device = device
        self.memory_bank = None
        self.scaler = StandardScaler()
    
    def _extract_features(self, pc):
        n_points = pc.shape[0]
        k_actual = min(self.k + 1, n_points)
        
        dists = torch.cdist(pc.unsqueeze(0), pc.unsqueeze(0)).squeeze(0)
        distances, indices = torch.topk(dists, k_actual, largest=False, dim=1)
        
        neighbor_idx = indices[:, 1:]
        neighbors = pc[neighbor_idx]
        centered = neighbors - pc.unsqueeze(1)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (k_actual - 1)
        
        try:
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = torch.sort(eigenvalues, dim=1, descending=True).values
            eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        except:
            eigenvalues = torch.ones(n_points, 3, device=pc.device) / 3
        
        # Only eigenvalues (curvature info)
        return eigenvalues
    
    def fit(self, point_clouds):
        all_features = []
        for pc in point_clouds:
            if isinstance(pc, np.ndarray):
                pc = torch.from_numpy(pc).float().to(self.device)
            features = self._extract_features(pc)
            all_features.append(features.cpu().numpy())
        
        all_features = np.vstack(all_features)
        all_features = np.nan_to_num(all_features, nan=0)
        all_features = self.scaler.fit_transform(all_features)
        
        if len(all_features) > self.memory_size:
            idx = np.random.choice(len(all_features), self.memory_size, replace=False)
            self.memory_bank = torch.from_numpy(all_features[idx]).float().to(self.device)
        else:
            self.memory_bank = torch.from_numpy(all_features).float().to(self.device)
        return self
    
    def score(self, point_cloud):
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float().to(self.device)
        
        features = self._extract_features(point_cloud)
        features_np = features.cpu().numpy()
        features_np = np.nan_to_num(features_np, nan=0)
        features_np = self.scaler.transform(features_np)
        features = torch.from_numpy(features_np).float().to(self.device)
        
        dists = torch.cdist(features, self.memory_bank)
        k = min(self.k_nn, self.memory_bank.shape[0])
        nearest_k = torch.topk(dists, k, largest=False, dim=1).values
        point_scores = nearest_k.mean(dim=1)
        
        point_scores_np = point_scores.cpu().numpy()
        threshold = np.percentile(point_scores_np, 95)
        top_scores = point_scores_np[point_scores_np >= threshold]
        
        return np.mean(top_scores) if len(top_scores) > 0 else np.mean(point_scores_np)


class GLAREBaseline:
    """GLARE with full features - single scale."""
    
    def __init__(self, k=16, memory_size=5000, k_nn=5, device='cuda'):
        self.k = k
        self.memory_size = memory_size
        self.k_nn = k_nn
        self.device = device
        self.memory_bank = None
        self.scaler = StandardScaler()
    
    def fit(self, point_clouds):
        all_features = []
        for pc in point_clouds:
            if isinstance(pc, np.ndarray):
                pc = torch.from_numpy(pc).float().to(self.device)
            features = extract_geometric_features_gpu(pc, self.k)
            all_features.append(features.cpu().numpy())
        
        all_features = np.vstack(all_features)
        all_features = np.nan_to_num(all_features, nan=0)
        all_features = self.scaler.fit_transform(all_features)
        
        if len(all_features) > self.memory_size:
            idx = np.random.choice(len(all_features), self.memory_size, replace=False)
            self.memory_bank = torch.from_numpy(all_features[idx]).float().to(self.device)
        else:
            self.memory_bank = torch.from_numpy(all_features).float().to(self.device)
        return self
    
    def score(self, point_cloud):
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float().to(self.device)
        
        features = extract_geometric_features_gpu(point_cloud, self.k)
        features_np = features.cpu().numpy()
        features_np = np.nan_to_num(features_np, nan=0)
        features_np = self.scaler.transform(features_np)
        features = torch.from_numpy(features_np).float().to(self.device)
        
        dists = torch.cdist(features, self.memory_bank)
        k = min(self.k_nn, self.memory_bank.shape[0])
        nearest_k = torch.topk(dists, k, largest=False, dim=1).values
        point_scores = nearest_k.mean(dim=1)
        
        point_scores_np = point_scores.cpu().numpy()
        threshold = np.percentile(point_scores_np, 95)
        top_scores = point_scores_np[point_scores_np >= threshold]
        
        return np.mean(top_scores) if len(top_scores) > 0 else np.mean(point_scores_np)


def evaluate_method(method_name, model_class, model_kwargs, data_dir, categories, n_shots=8, n_seeds=3):
    """Evaluate a method."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*50}")
    
    results = {}
    all_aurocs = []
    
    for cat in categories:
        cat_path = Path(data_dir) / cat
        train_dir = cat_path / 'train' / 'good' / 'xyz'
        
        if not train_dir.exists():
            continue
        
        train_files = sorted(train_dir.glob('*.tiff'))
        test_dir = cat_path / 'test'
        
        if not test_dir.exists():
            continue
        
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
            continue
        
        aurocs = []
        for seed in range(n_seeds):
            np.random.seed(seed + 42)
            torch.manual_seed(seed + 42)
            
            train_idx = np.random.choice(len(train_files), min(n_shots, len(train_files)), replace=False)
            train_pcs = []
            for idx in train_idx:
                pc = load_tiff_to_tensor(train_files[idx])
                if pc is not None:
                    train_pcs.append(pc)
            
            if not train_pcs:
                continue
            
            model = model_class(**model_kwargs)
            model.fit(train_pcs)
            
            scores, labels = [], []
            for f, label in zip(test_files, test_labels):
                pc = load_tiff_to_tensor(f)
                if pc is not None:
                    score = model.score(pc)
                    scores.append(score)
                    labels.append(label)
            
            if scores and len(set(labels)) > 1:
                try:
                    auc = roc_auc_score(labels, scores)
                    aurocs.append(auc)
                except:
                    pass
            
            torch.cuda.empty_cache()
        
        if aurocs:
            results[cat] = {
                'mean': float(np.mean(aurocs)),
                'std': float(np.std(aurocs)),
                'aurocs': aurocs
            }
            all_aurocs.extend(aurocs)
            print(f"  {cat}: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    
    return {
        'categories': results,
        'mean_auroc': float(np.mean(all_aurocs)) if all_aurocs else 0.0,
        'std_auroc': float(np.std(all_aurocs)) if all_aurocs else 0.0,
        'n_samples': len(all_aurocs)
    }


def main():
    """Main baseline comparison."""
    print("="*70)
    print("Round 40: Baseline Comparison Study")
    print("="*70)
    
    real3d_dir = Path('/home/cxs/桌面/aris2/Real3D-mvtec')
    results_dir = Path('/home/cxs/桌面/aris2/results')
    results_dir.mkdir(exist_ok=True)
    
    categories = ['candybar', 'car', 'chicken', 'diamond', 'duck',
                  'fish', 'gemstone', 'seahorse', 'starfish', 'toffees']
    
    methods = [
        ('Random', RandomBaseline, {}),
        ('Curvature Only (3 features)', CurvatureBaseline, {'k': 16}),
        ('GLARE (17 features, k=8)', GLAREBaseline, {'k': 8}),
        ('GLARE (17 features, k=16)', GLAREBaseline, {'k': 16}),
        ('GLARE (17 features, k=32)', GLAREBaseline, {'k': 32}),
    ]
    
    start_time = time.time()
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiment': 'Baseline Comparison Study (Round 40)',
        'categories_used': categories,
        'methods': {}
    }
    
    for name, model_class, kwargs in methods:
        method_results = evaluate_method(name, model_class, kwargs, real3d_dir, categories)
        results['methods'][name] = method_results
        print(f"\n{name}: {method_results['mean_auroc']*100:.2f}%")
    
    elapsed = time.time() - start_time
    results['elapsed_seconds'] = elapsed
    
    # Summary
    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)
    
    for name, data in results['methods'].items():
        print(f"{name:35}: {data['mean_auroc']*100:.2f}%")
    
    output_file = results_dir / 'round40_baseline_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
