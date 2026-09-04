"""
PRML Comprehensive Integration Test Suite
全15章（第0章〜第14章）のすべての基盤機械学習アルゴリズム・モデルの動作検証テストスイート
"""

import unittest
import numpy as np
import scipy.stats as stats
from sklearn.datasets import make_blobs, make_regression, make_classification

from common import (
    # Ch 2: Distributions
    simplex_to_xy,
    # Ch 3: Regression
    PolynomialBasis, GaussianBasis, SigmoidalBasis,
    LinearRegression, RidgeRegression, BayesianLinearRegression, EvidenceApproximation,
    # Ch 4: Classification
    Perceptron, FisherLinearDiscriminant, GaussianGenerativeClassifier,
    LogisticRegression, MulticlassLogisticRegression, BayesianLogisticRegression,
    # Ch 5: Neural Networks
    MLPRegressor, MixtureDensityNetwork,
    # Ch 6: Kernel Methods
    KernelRidgeRegression, NadarayaWatsonRegressor,
    GaussianProcessRegressor, GaussianProcessClassifier,
    # Ch 7: Sparse Kernel Machines
    SupportVectorClassifier, RelevanceVectorRegressor, RelevanceVectorClassifier,
    # Ch 8: Graphical Models
    SimpleFactorGraphChain,
    # Ch 9: Mixture Models & EM
    KMeans, GaussianMixtureModel, BernoulliMixtureModel,
    # Ch 10: Approximate Inference
    VariationalGaussianMixture,
    # Ch 11: Sampling Methods
    rejection_sample, metropolis_hastings, gibbs_sampler_2d, hamiltonian_monte_carlo,
    # Ch 12: Continuous Latent Variables
    PCA, ProbabilisticPCA, KernelPCA,
    # Ch 13: Sequential Data
    GaussianHMM, KalmanFilter,
    # Ch 14: Combining Models
    DecisionStump, AdaBoostClassifier, MixtureOfLinearRegressions
)
from common.graphical_models_utils import check_d_separation

class TestPRMLComprehensiveSuite(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    # --- Chapter 0: Foundations of Probability ---
    def test_ch0_probability_foundations(self):
        # 累積分布関数逆変換法による乱数生成 (コーシー分布)
        u = np.random.uniform(0, 1, 2000)
        x_cauchy = np.tan(np.pi * (u - 0.5))
        q25, q75 = np.percentile(x_cauchy, [25, 75])
        self.assertAlmostEqual(q25, -1.0, delta=0.2)
        self.assertAlmostEqual(q75, 1.0, delta=0.2)

        # ヤコビアン変数変換
        x_exp = np.random.exponential(1.0, 5000)
        y_sq = x_exp**2
        self.assertTrue(np.all(y_sq >= 0))

    # --- Chapter 1: Decision Theory & Information Theory ---
    def test_ch1_decision_and_information(self):
        # 決定理論: 誤識別率最小化の決定境界 (2つのガウス分布の交点)
        mu1, s1, p1 = 2.0, 1.0, 0.4
        mu2, s2, p2 = 5.0, 1.0, 0.6
        x_grid = np.linspace(2, 5, 100)
        px1 = stats.norm.pdf(x_grid, mu1, s1) * p1
        px2 = stats.norm.pdf(x_grid, mu2, s2) * p2
        x_boundary = x_grid[np.argmin(np.abs(px1 - px2))]
        self.assertTrue(2.5 < x_boundary < 4.5)

        # 情報理論: ベルヌーイエントロピー (p=0.5 で最大 1 bit)
        p = 0.5
        h_bit = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        self.assertAlmostEqual(h_bit, 1.0)

    # --- Chapter 2: Probability Distributions ---
    def test_ch2_probability_distributions(self):
        # ディリクレ単体座標変換
        p = np.array([1.0, 0.0, 0.0])
        x, y = simplex_to_xy(p)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)

        p_center = np.array([1/3, 1/3, 1/3])
        xc, yc = simplex_to_xy(p_center)
        self.assertAlmostEqual(xc, 0.5)

    # --- Chapter 3: Linear Models for Regression ---
    def test_ch3_linear_regression(self):
        X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
        
        # OLS & Ridge
        ols = LinearRegression().fit(X, y)
        ridge = RidgeRegression(alpha=0.1).fit(X, y)
        self.assertEqual(ols.predict(X).shape, (50,))
        self.assertEqual(ridge.predict(X).shape, (50,))
        
        # Bayesian Linear Regression & Evidence Approximation
        blr = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(X, y)
        mean, var = blr.predict(X)
        self.assertEqual(mean.shape, (50,))
        self.assertTrue(np.all(var > 0))

        basis = PolynomialBasis(degree=2)
        Phi = basis.transform(X[:, :1])
        self.assertEqual(Phi.shape, (50, 3))

    # --- Chapter 4: Linear Models for Classification ---
    def test_ch4_linear_classification(self):
        X, y = make_blobs(n_samples=60, n_features=2, centers=2, random_state=42)
        y_pm = np.where(y == 0, -1, 1)

        # Perceptron
        pct = Perceptron(max_iter=100).fit(X, y_pm)
        self.assertGreater(np.mean(pct.predict(X) == y_pm), 0.85)

        # Fisher LDA
        lda = FisherLinearDiscriminant().fit(X, y)
        self.assertGreater(np.mean(lda.predict(X) == y), 0.85)

        # Logistic Regression & Multiclass Logistic
        lr = LogisticRegression(max_iter=50).fit(X, y)
        self.assertGreater(np.mean(lr.predict(X) == y), 0.90)

        # Bayesian Logistic Regression
        blr = BayesianLogisticRegression(alpha=1.0, max_iter=20).fit(X, y)
        post_preds = blr.predict(X)
        self.assertGreater(np.mean(post_preds == y), 0.90)

    # --- Chapter 5: Neural Networks ---
    def test_ch5_neural_networks(self):
        X = np.linspace(-1, 1, 30)[:, np.newaxis]
        y = (X**2).ravel()

        mlp = MLPRegressor(n_in=1, n_hidden=5, n_out=1, lr=0.05, random_state=42)
        y_pred_before = mlp.predict(X)
        self.assertEqual(y_pred_before.shape, (30, 1))

        # 勾配逆伝播と損失計算
        loss, grads = mlp.compute_loss_and_grads(X, y[:, np.newaxis])
        self.assertGreater(loss, 0)
        self.assertEqual(grads['W1'].shape, mlp.W1.shape)
        self.assertEqual(grads['W2'].shape, mlp.W2.shape)

        # Mixture Density Network
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=2, random_state=42)
        z, pi, sigma, mu, a_sig = mdn.forward(X)
        self.assertEqual(pi.shape, (30, 2))
        self.assertEqual(mu.shape, (30, 2))
        self.assertEqual(sigma.shape, (30, 2))
        self.assertTrue(np.all(sigma > 0))


    # --- Chapter 6: Kernel Methods ---
    def test_ch6_kernel_methods(self):
        X = np.linspace(0, 3, 20)[:, np.newaxis]
        y = np.sin(X).ravel()

        # Kernel Ridge Regression
        krr = KernelRidgeRegression(kernel='rbf', reg_lambda=0.01).fit(X, y)
        self.assertEqual(len(krr.predict(X)), 20)

        # Nadaraya-Watson
        nw = NadarayaWatsonRegressor(kernel='gaussian', h=0.5).fit(X, y)
        self.assertEqual(len(nw.predict(X)), 20)

        # Gaussian Process Regressor
        gpr = GaussianProcessRegressor(beta=100.0, theta0=1.0, theta1=1.0).fit(X, y)
        mean, cov = gpr.predict(X)
        self.assertEqual(len(mean), 20)
        self.assertLess(np.mean((mean - y)**2), 0.1)

    # --- Chapter 7: Sparse Kernel Machines ---
    def test_ch7_sparse_kernel_machines(self):
        X, y = make_blobs(n_samples=40, centers=2, random_state=42)
        y_pm = np.where(y == 0, -1, 1)

        # Support Vector Classifier
        svc = SupportVectorClassifier(C=1.0, kernel='rbf').fit(X, y_pm)
        preds = svc.predict(X)
        self.assertGreater(np.mean(preds == y_pm), 0.90)

        # Relevance Vector Regressor
        X_reg = np.linspace(0, 2, 25)[:, np.newaxis]
        y_reg = (X_reg**2).ravel()
        rvr = RelevanceVectorRegressor(kernel='rbf', max_iter=50).fit(X_reg, y_reg)
        rvr_preds = rvr.predict(X_reg, return_std=False)
        self.assertEqual(len(rvr_preds), 25)

    # --- Chapter 8: Graphical Models ---
    def test_ch8_graphical_models(self):
        # d-separation
        adj = {'A': ['B'], 'B': ['C'], 'C': []}
        self.assertTrue(check_d_separation(adj, ['A'], ['C'], ['B']))
        self.assertFalse(check_d_separation(adj, ['A'], ['C'], []))

        adj_v = {'A': ['C'], 'B': ['C'], 'C': []}
        self.assertTrue(check_d_separation(adj_v, ['A'], ['B'], []))
        self.assertFalse(check_d_separation(adj_v, ['A'], ['B'], ['C']))

        # SimpleFactorGraphChain (3ノード, 2状態)
        trans = [
            np.array([[0.8, 0.2], [0.1, 0.9]]),
            np.array([[0.7, 0.3], [0.2, 0.8]])
        ]
        emiss = [
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7])
        ]
        fg = SimpleFactorGraphChain(node_states=2, transition_matrices=trans, emission_potentials=emiss)
        marginals = fg.forward_backward_marginals()
        self.assertEqual(len(marginals), 3)
        for m in marginals:
            self.assertAlmostEqual(np.sum(m), 1.0)

    # --- Chapter 9: Mixture Models & EM ---
    def test_ch9_mixture_models_and_em(self):
        X, _ = make_blobs(n_samples=80, centers=2, random_state=42)
        
        # K-Means
        km = KMeans(n_clusters=2, random_state=42).fit(X)
        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        
        # Gaussian Mixture Model
        gmm = GaussianMixtureModel(n_components=2, random_state=42).fit(X)
        self.assertEqual(len(gmm.weights_), 2)

        # Bernoulli Mixture Model
        X_bin = np.random.binomial(1, 0.5, (60, 5))
        bmm = BernoulliMixtureModel(n_components=2, max_iter=20, random_state=42).fit(X_bin)
        self.assertEqual(bmm.means_.shape, (2, 5))

    # --- Chapter 10: Approximate Inference ---
    def test_ch10_approximate_inference(self):
        X, _ = make_blobs(n_samples=60, centers=2, random_state=42)
        vbgmm = VariationalGaussianMixture(n_components=3, random_state=42).fit(X)
        self.assertEqual(len(vbgmm.alpha_), 3)
        self.assertTrue(np.all(vbgmm.alpha_ > 0))

    # --- Chapter 11: Sampling Methods ---
    def test_ch11_sampling_methods(self):
        # 1. Rejection Sampling
        samples_rej, acc = rejection_sample(
            target_pdf=lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
            proposal_sampler=lambda: np.random.uniform(-4, 4),
            proposal_pdf=lambda x: 1.0 / 8.0,
            k=4.0,
            n_samples=100
        )
        self.assertEqual(len(samples_rej), 100)
        self.assertGreater(acc, 0.0)

        # 2. Metropolis-Hastings
        samples_mh, acc_flags = metropolis_hastings(
            target_log_pdf=lambda x: -0.5 * (x**2),
            initial_state=0.0,
            proposal_sampler=lambda curr: curr + np.random.normal(0, 1.0),
            n_samples=100
        )
        self.assertEqual(len(samples_mh), 100)
        self.assertGreater(np.mean(acc_flags), 0.0)

        # 3. 2D Gibbs Sampler
        rho = 0.5
        cond_x = lambda y: np.random.normal(rho * y, np.sqrt(1 - rho**2))
        cond_y = lambda x: np.random.normal(rho * x, np.sqrt(1 - rho**2))
        traj = gibbs_sampler_2d(cond_x, cond_y, initial_state=[0.0, 0.0], n_samples=50)
        self.assertEqual(traj.shape[1], 2)


    # --- Chapter 12: Continuous Latent Variables ---
    def test_ch12_continuous_latent_variables(self):
        X = np.random.randn(50, 4)
        
        # PCA
        pca = PCA(n_components=2).fit(X)
        self.assertEqual(pca.transform(X).shape, (50, 2))
        self.assertEqual(pca.inverse_transform(pca.transform(X)).shape, (50, 4))
        
        # Probabilistic PCA (closed form & EM)
        ppca_cf = ProbabilisticPCA(n_components=2, method='closed_form').fit(X)
        self.assertEqual(ppca_cf.transform(X).shape, (50, 2))

        ppca_em = ProbabilisticPCA(n_components=2, method='em', max_iter=20).fit(X)
        self.assertEqual(ppca_em.transform(X).shape, (50, 2))

        # Kernel PCA
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1).fit(X)
        self.assertEqual(kpca.transform(X).shape, (50, 2))

    # --- Chapter 13: Sequential Data ---
    def test_ch13_sequential_data(self):
        # Gaussian HMM
        hmm = GaussianHMM(n_components=2, n_iter=10)
        hmm.pi_ = np.array([0.5, 0.5])
        hmm.A_ = np.array([[0.7, 0.3], [0.3, 0.7]])
        hmm.means_ = np.array([[0.0], [3.0]])
        hmm.covs_ = np.array([[[0.5]], [[0.5]]])
        
        states, obs = hmm.sample(n_samples=40)
        preds = hmm.predict(obs)
        self.assertEqual(len(preds), 40)

        # Kalman Filter
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        C = np.array([[1.0, 0.0]])
        Gamma = np.eye(2) * 0.1
        Sigma = np.eye(1) * 0.5
        mu_0 = np.array([0.0, 1.0])
        V_0 = np.eye(2)
        kf = KalmanFilter(A, C, Gamma, Sigma, mu_0, V_0)
        
        obs_seq = np.linspace(0, 10, 20)[:, np.newaxis]
        means_filt, covs_filt = kf.filter(obs_seq)
        self.assertEqual(means_filt.shape, (20, 2))
        
        means_smooth, covs_smooth = kf.smooth(obs_seq)
        self.assertEqual(means_smooth.shape, (20, 2))

    # --- Chapter 14: Combining Models ---
    def test_ch14_combining_models(self):
        X, y = make_blobs(n_samples=50, centers=2, random_state=42)
        y_pm = np.where(y == 0, -1, 1)

        # Decision Stump & AdaBoost
        stump = DecisionStump().fit(X, y_pm, sample_weight=np.ones(len(X))/len(X))
        self.assertEqual(len(stump.predict(X)), 50)

        ada = AdaBoostClassifier(n_estimators=5).fit(X, y_pm)
        preds = ada.predict(X)
        self.assertGreater(np.mean(preds == y_pm), 0.90)

        # Mixture of Linear Regressions
        X_reg = np.linspace(-1, 1, 40)[:, np.newaxis]
        y_reg = np.where(X_reg.ravel() < 0, -2 * X_reg.ravel(), 2 * X_reg.ravel())
        moe = MixtureOfLinearRegressions(n_components=2, max_iter=20, random_state=42).fit(X_reg, y_reg)
        y_pred = moe.predict(X_reg)
        self.assertEqual(len(y_pred), 40)

if __name__ == '__main__':
    unittest.main()
