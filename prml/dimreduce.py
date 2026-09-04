"""
PRML Dimensionality Reduction Module (Chapter 12)
主成分分析 (PCA), 確率的主成分分析 (Probabilistic PCA), カーネルPCA (Kernel PCA)
"""

from common.pca_ppca_utils import (
    PCA,
    ProbabilisticPCA,
    KernelPCA,
)

__all__ = [
    "PCA",
    "ProbabilisticPCA",
    "KernelPCA",
]
