"""GLARE: Geometry-based Local Anomaly Recognition and Encoding.

Training-free, depth-only few-shot 3D industrial anomaly detection.

Modules
-------
glare_dams  : main pipeline (12D descriptor + Density-Aware Memory Scoring + multi-seed eval)
glare_core  : GPU-accelerated core (feature extraction + memory bank scoring)
glare_pro   : extended / configurable variant
"""

__version__ = "1.0.0"
