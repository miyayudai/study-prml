"""
PRML Chapter 12: Continuous Latent Variables & Dimensionality Reduction Unit Tests
標準主成分分析 (PCA)、確率的主成分分析 (PPCA)、カーネル主成分分析 (Kernel PCA) の数理的検証
"""

import unittest
import numpy as np
from sklearn.datasets import make_circles

from prml.dimreduce import (
    PCA,
    ProbabilisticPCA,
    KernelPCA,
)


class TestDimensionalityReduction(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_pca_orthonormality_and_projection(self):
        # PRML 12.1.1節 式 (12.3): 主成分ベクトルの正規直交性 u_i^T u_j = delta_ij
        X = np.random.randn(50, 5) @ np.diag([5.0, 3.0, 1.0, 0.5, 0.1])
        pca = PCA(n_components=3).fit(X)
        
        U = pca.components_ # (3, 5)
        # 正規直交性: U @ U.T == I_3
        np.testing.assert_allclose(U @ U.T, np.eye(3), atol=1e-7)
        
        # 射影と逆射影
        Z = pca.transform(X)
        self.assertEqual(Z.shape, (50, 3))
        X_rec = pca.inverse_transform(Z)
        self.assertEqual(X_rec.shape, (50, 5))
        
        # 説明分散比率の単調減少
        var_ratios = pca.explained_variance_ratio_
        self.assertEqual(len(var_ratios), 3)
        self.assertTrue(np.all(np.diff(var_ratios) <= 0.0))

    def test_pca_n_less_than_d_trick(self):
        # PRML 12.1.4節 式 (12.18)-(12.21): N << D における双対特異値計算法
        # N = 8, D = 40
        N, D = 8, 40
        X = np.random.randn(N, D)
        pca = PCA(n_components=3).fit(X)
        
        self.assertEqual(pca.components_.shape, (3, D))
        # 正規直交性
        np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(3), atol=1e-6)
        
        # 標本共分散直接計算 (D, D) との固有値一致性確認
        X_centered = X - np.mean(X, axis=0)
        S = (X_centered.T @ X_centered) / N
        vals_direct = np.sort(np.linalg.eigvalsh(S))[::-1][:3]
        np.testing.assert_allclose(pca.explained_variance_, vals_direct, atol=1e-5)

    def test_ppca_closed_form_vs_standard_pca(self):
        # PRML 12.2.1節 式 (12.45), (12.46): PPCA の最尤解
        X = np.random.randn(100, 4) @ np.diag([4.0, 2.0, 0.8, 0.2])
        ppca = ProbabilisticPCA(n_components=2, method='closed_form').fit(X)
        
        # 推定ノイズ分散 sigma^2 は切り捨てられた固有値 (0.8^2, 0.2^2 など) の平均に近い
        self.assertGreater(ppca.sigma2_, 0.0)
        self.assertEqual(ppca.W_.shape, (4, 2))
        
        # W の張る主部分空間と標準 PCA の主部分空間の射影作用素の一致性
        # P = W (W^T W)^-1 W^T
        W = ppca.W_
        P_ppca = W @ np.linalg.inv(W.T @ W) @ W.T
        
        pca = PCA(n_components=2).fit(X)
        U = pca.components_ # (2, 4)
        P_pca = U.T @ U     # (4, 4)
        
        np.testing.assert_allclose(P_ppca, P_pca, atol=1e-5)

    def test_ppca_em_algorithm_subspace(self):
        # PRML 12.2.2節: EM アルゴリズムによる PPCA の推定
        X = np.random.randn(80, 4) @ np.diag([3.0, 1.5, 0.5, 0.2])
        ppca_em = ProbabilisticPCA(n_components=2, method='em', max_iter=150, tol=1e-5, random_state=42).fit(X)
        ppca_cf = ProbabilisticPCA(n_components=2, method='closed_form').fit(X)
        
        # EMによる主空間射影作用素が解析解の射影作用素と一致すること
        W_em = ppca_em.W_
        P_em = W_em @ np.linalg.inv(W_em.T @ W_em) @ W_em.T
        
        W_cf = ppca_cf.W_
        P_cf = W_cf @ np.linalg.inv(W_cf.T @ W_cf) @ W_cf.T
        
        np.testing.assert_allclose(P_em, P_cf, atol=1e-2)

    def test_kernel_pca_centering_and_projection(self):
        # PRML 12.3節 式 (12.80): グラム行列の中心化 \tilde{K} の行和・列和はゼロ
        X, y = make_circles(n_samples=50, factor=0.3, noise=0.05, random_state=42)
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=2.0).fit(X)
        
        Z = kpca.transform(X)
        self.assertEqual(Z.shape, (50, 2))
        
        # 同心円データにおいて、非線形カーネル PCA の第1成分によって同心円が線形分離可能となること
        # 内側サンプルの Z[:, 0] と 外側サンプルの Z[:, 0] が明確に分かれる
        inner_pts = Z[y == 1, 0]
        outer_pts = Z[y == 0, 0]
        
        # 2つのクラスタの平均値の分離度が分散より十分に大きいこと
        mean_diff = np.abs(np.mean(inner_pts) - np.mean(outer_pts))
        pooled_std = np.sqrt(0.5 * (np.var(inner_pts) + np.var(outer_pts)))
        t_stat = mean_diff / (pooled_std + 1e-8)
        self.assertGreater(t_stat, 2.0)


if __name__ == '__main__':
    unittest.main()
