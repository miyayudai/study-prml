"""
PRML Comprehensive Integration Test Suite
全章のアルゴリズムとモデルが scikit-learn API に準拠して正常に動作するかを検証
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_regression, make_circles

from common import (
    LinearRegression, RidgeRegression, BayesianLinearRegression, EvidenceApproximation,
    LogisticRegression, MulticlassLogisticRegression,
    GaussianProcessRegressor,
    SupportVectorClassifier, RelevanceVectorRegressor,
    KMeans, GaussianMixtureModel,
    VariationalGaussianMixture,
    rejection_sample, metropolis_hastings,
    PCA, ProbabilisticPCA, KernelPCA,
    GaussianHMM, KalmanFilter,
    AdaBoostClassifier, MixtureOfLinearRegressions
)

class TestPRMLComprehensiveSuite(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_ch3_linear_models_for_regression(self):
        X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
        
        # OLS & Ridge
        ols = LinearRegression().fit(X, y)
        ridge = RidgeRegression(alpha=0.1).fit(X, y)
        self.assertEqual(ols.predict(X).shape, (50,))
        self.assertEqual(ridge.predict(X).shape, (50,))
        
        # Bayesian Linear Regression
        blr = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(X, y)
        mean, var = blr.predict(X)
        self.assertEqual(mean.shape, (50,))
        self.assertTrue(np.all(var > 0))

    def test_ch4_linear_models_for_classification(self):
        X, y = make_blobs(n_samples=60, n_features=2, centers=2, random_state=42)
        lr = LogisticRegression(max_iter=50).fit(X, y)
        preds = lr.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)

    def test_ch6_kernel_methods(self):
        X = np.linspace(0, 5, 20)[:, np.newaxis]
        y = np.sin(X).ravel()
        gpr = GaussianProcessRegressor(beta=100.0, theta0=1.0, theta1=1.0).fit(X, y)
        mean, cov = gpr.predict(X)
        self.assertEqual(len(mean), 20)
        self.assertLess(np.mean((mean - y)**2), 0.1)

    def test_ch7_sparse_kernel_machines(self):
        X, y = make_blobs(n_samples=40, centers=2, random_state=42)
        y = np.where(y == 0, -1, 1)
        svc = SupportVectorClassifier(C=1.0).fit(X, y)
        preds = svc.predict(X)
        self.assertGreater(np.mean(preds == y), 0.90)

    def test_ch9_mixture_models_and_em(self):
        X, _ = make_blobs(n_samples=80, centers=2, random_state=42)
        km = KMeans(n_clusters=2, random_state=42).fit(X)
        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        
        gmm = GaussianMixtureModel(n_components=2, random_state=42).fit(X)
        self.assertEqual(len(gmm.weights_), 2)

    def test_ch10_approximate_inference(self):
        X, _ = make_blobs(n_samples=60, centers=2, random_state=42)
        vbgmm = VariationalGaussianMixture(n_components=3, random_state=42).fit(X)
        self.assertEqual(len(vbgmm.alpha_), 3)

    def test_ch11_sampling_methods(self):
        # 1D ガウスからの Rejection Sampling
        samples, accept_rate = rejection_sample(
            target_pdf=lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
            proposal_sampler=lambda: np.random.uniform(-4, 4),
            proposal_pdf=lambda x: 1.0 / 8.0,
            k=4.0,
            n_samples=100
        )
        self.assertEqual(len(samples), 100)
        self.assertGreater(accept_rate, 0.0)

    def test_ch12_continuous_latent_variables(self):
        X = np.random.randn(50, 4)
        pca = PCA(n_components=2).fit(X)
        self.assertEqual(pca.transform(X).shape, (50, 2))
        
        ppca = ProbabilisticPCA(n_components=2, method='closed_form').fit(X)
        self.assertEqual(ppca.transform(X).shape, (50, 2))

    def test_ch13_sequential_data(self):
        hmm = GaussianHMM(n_components=2, n_iter=10)
        hmm.pi_ = np.array([0.5, 0.5])
        hmm.A_ = np.array([[0.7, 0.3], [0.3, 0.7]])
        hmm.means_ = np.array([[0.0], [3.0]])
        hmm.covs_ = np.array([[[0.5]], [[0.5]]])
        
        states, obs = hmm.sample(n_samples=40)
        preds = hmm.predict(obs)
        self.assertEqual(len(preds), 40)

    def test_ch14_combining_models(self):
        X, y = make_blobs(n_samples=50, centers=2, random_state=42)
        y = np.where(y == 0, -1, 1)
        ada = AdaBoostClassifier(n_estimators=5).fit(X, y)
        preds = ada.predict(X)
        self.assertGreater(np.mean(preds == y), 0.90)

if __name__ == '__main__':
    unittest.main()
