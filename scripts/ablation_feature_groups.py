#!/usr/bin/env python3
"""
Round 41: Feature Group Ablation Study

Addresses Reviewer Weakness 1: "Why these 17 features?"

Feature Groups:
1. Eigenvalue-based shape (7): linearity, planarity, sphericity, anisotropy, omnivariance, eigen-entropy, curvature
2. Distance statistics (5): mean, std, max, min, median of neighbor distances
3. Height statistics (5): z_mean, z_std, z_range, z_min_rel, z_max_rel

Tests:
- Shape only (7D)
- Distance only (5D)
- Height only (5D)
- Shape + Distance (12D)
- Shape + Height (12D)
- Distance + Height (10D)
- Full (17D)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FeatureGroupAblation:
    """Feature group ablation for GLARE"""
    
    def __init__(self, k=16, device='cuda'):
        self.k = k
        self.device = device
        
    def extract_all_features(self, points):
        """Extract all 17 features and return grouped"""
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float().to(self.device)
        
        n_points = points.shape[0]
        
        # Build KNN graph
        dists = torch.cdist(points, points)
        _, indices = torch.topk(dists, self.k + 1, dim=1, largest=False)
        indices = indices[:, 1:]  # exclude self
        
        # Get neighbor points and distances
        neighbors = points[indices]  # [N, k, 3]
        neighbor_dists = torch.gather(dists, 1, indices)  # [N, k]
        
        # === GROUP 1: Eigenvalue-based shape features (7D) ===
        centered = neighbors - points.unsqueeze(1)  # [N, k, 3]
        cov = torch.bmm(centered.transpose(1, 2), centered) / self.k  # [N, 3, 3]
        
        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov)  # [N, 3]
        eigenvalues = torch.sort(eigenvalues, dim=1, descending=True)[0]  # [N, 3]
        l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        
        eps = 1e-8
        sum_eig = l1 + l2 + l3 + eps
        
        # Shape features
        linearity = (l1 - l2) / (l1 + eps)
        planarity = (l2 - l3) / (l1 + eps)
        sphericity = l3 / (l1 + eps)
        anisotropy = (l1 - l3) / (l1 + eps)
        omnivariance = torch.pow(l1 * l2 * l3 + eps, 1/3)
        
        # Eigen-entropy
        norm_eig = eigenvalues / sum_eig.unsqueeze(1)
        eigen_entropy = -torch.sum(norm_eig * torch.log(norm_eig + eps), dim=1)
        
        # Curvature
        curvature = l3 / sum_eig
        
        shape_features = torch.stack([
            linearity, planarity, sphericity, anisotropy,
            omnivariance, eigen_entropy, curvature
        ], dim=1)  # [N, 7]
        
        # === GROUP 2: Distance statistics (5D) ===
        dist_mean = neighbor_dists.mean(dim=1)
        dist_std = neighbor_dists.std(dim=1)
        dist_max = neighbor_dists.max(dim=1)[0]
        dist_min = neighbor_dists.min(dim=1)[0]
        dist_median = neighbor_dists.median(dim=1)[0]
        
        distance_features = torch.stack([
            dist_mean, dist_std, dist_max, dist_min, dist_median
        ], dim=1)  # [N, 5]
        
        # === GROUP 3: Height statistics (5D) ===
        z_coords = neighbors[:, :, 2]  # [N, k]
        z_mean = z_coords.mean(dim=1)
        z_std = z_coords.std(dim=1)
        z_range = z_coords.max(dim=1)[0] - z_coords.min(dim=1)[0]
        z_min_rel = points[:, 2] - z_coords.min(dim=1)[0]
        z_max_rel = z_coords.max(dim=1)[0] - points[:, 2]
        
        height_features = torch.stack([
            z_mean, z_std, z_range, z_min_rel, z_max_rel
        ], dim=1)  # [N, 5]
        
        return {
            'shape': shape_features,
            'distance': distance_features,
            'height': height_features
        }
    
    def combine_groups(self, feature_dict, groups):
        """Combine specified feature groups"""
        parts = []
        for g in groups:
            parts.append(feature_dict[g])
        return torch.cat(parts, dim=1)
    
    def run_ablation(self, train_features, test_features, test_labels):
        """Run anomaly detection with given features"""
        # Normalize
        mean = train_features.mean(dim=0, keepdim=True)
        std = train_features.std(dim=0, keepdim=True) + 1e-8
        train_norm = (train_features - mean) / std
        test_norm = (test_features - mean) / std
        
        # Memory bank (subsample for efficiency)
        if train_norm.shape[0] > 5000:
            indices = torch.randperm(train_norm.shape[0])[:5000]
            memory_bank = train_norm[indices]
        else:
            memory_bank = train_norm
        
        # Compute distances
        dists = torch.cdist(test_norm, memory_bank)
        min_dists = dists.min(dim=1)[0]
        
        # Aggregate to sample-level scores (assuming 2048 points per sample)
        n_samples = test_labels.shape[0]
        n_points = min_dists.shape[0]
        points_per_sample = n_points // n_samples
        
        scores = []
        for i in range(n_samples):
            start = i * points_per_sample
            end = (i + 1) * points_per_sample
            sample_dists = min_dists[start:end]
            # Top-95 percentile
            score = torch.quantile(sample_dists, 0.95)
            scores.append(score.item())
        
        scores = np.array(scores)
        test_labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels
        
        try:
            auroc = roc_auc_score(test_labels_np, scores)
        except:
            auroc = 0.5
            
        return auroc


def load_real3d_category(category, base_path, n_shots=8, device='cuda'):
    """Load a Real3D-AD category with correct path structure"""
    import tifffile
    
    cat_path = Path(base_path) / category
    
    def load_point_cloud(tiff_path, n_points=2048):
        try:
            data = tifffile.imread(str(tiff_path))
            
            # Handle different TIFF formats
            if data.ndim == 3 and data.shape[2] == 3:
                # XYZ format: (H, W, 3)
                points = data.reshape(-1, 3)
                valid = ~np.any(np.isnan(points), axis=1) & ~np.all(points == 0, axis=1)
                points = points[valid].astype(np.float32)
            elif data.ndim == 2:
                # Depth map format
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
            elif len(points) < n_points:
                idx = np.random.choice(len(points), n_points, replace=True)
                points = points[idx]
            
            return points
        except Exception as e:
            return None
    
    # Load train - path is: category/train/good/xyz/*.tiff
    train_xyz_path = cat_path / "train" / "good" / "xyz"
    train_files = sorted(train_xyz_path.glob("*.tiff"))[:n_shots]
    train_points = []
    for f in train_files:
        pc = load_point_cloud(f)
        if pc is not None:
            train_points.append(pc)
    
    if len(train_points) == 0:
        return None, None, None
    train_points = np.stack(train_points)
    
    # Load test - path is: category/test/{good,anomaly_type}/xyz/*.tiff
    test_base = cat_path / "test"
    test_points = []
    test_labels = []
    
    # Good samples
    test_good_xyz = test_base / "good" / "xyz"
    if test_good_xyz.exists():
        good_files = sorted(test_good_xyz.glob("*.tiff"))[:10]
        for f in good_files:
            pc = load_point_cloud(f)
            if pc is not None:
                test_points.append(pc)
                test_labels.append(0)
    
    # Anomaly samples - can be in various subdirectories
    for subdir in test_base.iterdir():
        if subdir.is_dir() and subdir.name != "good":
            anomaly_xyz = subdir / "xyz"
            if anomaly_xyz.exists():
                anomaly_files = sorted(anomaly_xyz.glob("*.tiff"))[:5]  # limit per type
                for f in anomaly_files:
                    pc = load_point_cloud(f)
                    if pc is not None:
                        test_points.append(pc)
                        test_labels.append(1)
    
    if len(test_points) == 0:
        return None, None, None
    
    test_points = np.stack(test_points)
    test_labels = np.array(test_labels)
    
    return train_points, test_points, test_labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    base_path = Path("/home/cxs/桌面/aris2/Real3D-mvtec")
    
    # Categories
    categories = ['candybar', 'diamond', 'duck', 'fish', 'gemstone',
                  'seahorse', 'shell', 'starfish', 'toffees', 'car']
    
    # Feature group configurations
    configs = {
        'Shape only (7D)': ['shape'],
        'Distance only (5D)': ['distance'],
        'Height only (5D)': ['height'],
        'Shape+Distance (12D)': ['shape', 'distance'],
        'Shape+Height (12D)': ['shape', 'height'],
        'Distance+Height (10D)': ['distance', 'height'],
        'Full GLARE (17D)': ['shape', 'distance', 'height']
    }
    
    seeds = [42, 123, 456]
    results = {config: {cat: [] for cat in categories} for config in configs}
    
    ablation = FeatureGroupAblation(k=16, device=device)
    
    print("=" * 70)
    print("Round 41: Feature Group Ablation Study")
    print("=" * 70)
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        
        for cat in categories:
            print(f"\nCategory: {cat}")
            
            # Load data
            train_points, test_points, test_labels = load_real3d_category(
                cat, base_path, n_shots=8, device=device
            )
            
            if train_points is None:
                print(f"  Skipping {cat} - data not found")
                continue
            
            # Extract all features for train
            train_features_grouped = {}
            for tp in train_points:
                tp_tensor = torch.from_numpy(tp).float().to(device)
                fg = ablation.extract_all_features(tp_tensor)
                for key in fg:
                    if key not in train_features_grouped:
                        train_features_grouped[key] = []
                    train_features_grouped[key].append(fg[key])
            
            for key in train_features_grouped:
                train_features_grouped[key] = torch.cat(train_features_grouped[key], dim=0)
            
            # Extract all features for test
            test_features_grouped = {}
            for tp in test_points:
                tp_tensor = torch.from_numpy(tp).float().to(device)
                fg = ablation.extract_all_features(tp_tensor)
                for key in fg:
                    if key not in test_features_grouped:
                        test_features_grouped[key] = []
                    test_features_grouped[key].append(fg[key])
            
            for key in test_features_grouped:
                test_features_grouped[key] = torch.cat(test_features_grouped[key], dim=0)
            
            test_labels_tensor = torch.from_numpy(test_labels).long().to(device)
            
            # Run each configuration
            for config_name, groups in configs.items():
                train_feats = ablation.combine_groups(train_features_grouped, groups)
                test_feats = ablation.combine_groups(test_features_grouped, groups)
                
                auroc = ablation.run_ablation(train_feats, test_feats, test_labels_tensor)
                results[config_name][cat].append(auroc)
                
                print(f"  {config_name}: {auroc*100:.2f}%")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("FEATURE GROUP ABLATION RESULTS (Mean ± Std)")
    print("=" * 70)
    
    summary = {}
    for config_name in configs:
        all_scores = []
        for cat in categories:
            all_scores.extend(results[config_name][cat])
        mean_score = np.mean(all_scores) * 100
        std_score = np.std(all_scores) * 100
        summary[config_name] = {'mean': mean_score, 'std': std_score, 'scores': all_scores}
        print(f"{config_name}: {mean_score:.2f}% ± {std_score:.2f}%")
    
    # Statistical comparisons
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISONS (vs Full GLARE)")
    print("=" * 70)
    
    full_scores = summary['Full GLARE (17D)']['scores']
    for config_name in configs:
        if config_name == 'Full GLARE (17D)':
            continue
        config_scores = summary[config_name]['scores']
        t_stat, p_value = stats.ttest_rel(full_scores, config_scores)
        diff = summary['Full GLARE (17D)']['mean'] - summary[config_name]['mean']
        print(f"Full vs {config_name}: Δ={diff:+.2f}%, p={p_value:.4f}")
    
    # Feature importance ranking
    print("\n" + "=" * 70)
    print("FEATURE GROUP IMPORTANCE RANKING")
    print("=" * 70)
    
    # Calculate contribution of each group
    full_mean = summary['Full GLARE (17D)']['mean']
    
    shape_contrib = full_mean - summary['Distance+Height (10D)']['mean']
    distance_contrib = full_mean - summary['Shape+Height (12D)']['mean']
    height_contrib = full_mean - summary['Shape+Distance (12D)']['mean']
    
    print(f"Shape features (7D): +{shape_contrib:.2f}% contribution")
    print(f"Distance features (5D): +{distance_contrib:.2f}% contribution")
    print(f"Height features (5D): +{height_contrib:.2f}% contribution")
    
    # Save results
    output = {
        'timestamp': str(np.datetime64('now')),
        'experiment': 'Round 41: Feature Group Ablation',
        'device': device,
        'categories': categories,
        'seeds': seeds,
        'configs': {k: v for k, v in configs.items()},
        'results': {config: {cat: results[config][cat] for cat in categories} for config in configs},
        'summary': {config: {'mean': summary[config]['mean'], 'std': summary[config]['std']} for config in configs},
        'feature_contributions': {
            'shape': shape_contrib,
            'distance': distance_contrib,
            'height': height_contrib
        }
    }
    
    output_path = Path("/home/cxs/桌面/aris2/results/round41_feature_group_ablation.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output


if __name__ == "__main__":
    main()
