"""
PRML Package Architecture & API Integrity Unit Tests
すべてのサブパッケージ (prml.*) のインポート整合性とパブリックインターフェース検証
"""

import unittest
import inspect

import prml
import prml.linear as linear
import prml.kernel as kernel
import prml.nn as nn
import prml.clustering as clustering
import prml.sampling as sampling
import prml.dimreduce as dimreduce
import prml.sequential as sequential
import prml.ensemble as ensemble
import prml.graphical as graphical
import prml.distributions as distributions
import prml.variational as variational

class TestPRMLPackageAPI(unittest.TestCase):

    def test_package_metadata(self):
        self.assertTrue(hasattr(prml, '__version__'))
        self.assertTrue(hasattr(prml, '__author__'))
        self.assertEqual(prml.__author__, "miyayudai")

    def test_linear_exports(self):
        expected = [
            "LinearRegression", "RidgeRegression", "BayesianLinearRegression",
            "Perceptron", "FisherLinearDiscriminant", "LogisticRegression"
        ]
        for name in expected:
            self.assertTrue(hasattr(linear, name), f"Missing {name} in prml.linear")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_kernel_exports(self):
        expected = [
            "KernelRidgeRegression", "GaussianProcessRegressor",
            "SupportVectorClassifier", "RelevanceVectorRegressor"
        ]
        for name in expected:
            self.assertTrue(hasattr(kernel, name), f"Missing {name} in prml.kernel")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_clustering_exports(self):
        expected = [
            "KMeans", "GaussianMixtureModel", "BernoulliMixtureModel", "VariationalGaussianMixture"
        ]
        for name in expected:
            self.assertTrue(hasattr(clustering, name), f"Missing {name} in prml.clustering")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_sampling_exports(self):
        expected = [
            "rejection_sample", "metropolis_hastings", "gibbs_sampler_2d", "hamiltonian_monte_carlo"
        ]
        for name in expected:
            self.assertTrue(hasattr(sampling, name), f"Missing {name} in prml.sampling")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_dimreduce_exports(self):
        expected = ["PCA", "ProbabilisticPCA", "KernelPCA"]
        for name in expected:
            self.assertTrue(hasattr(dimreduce, name), f"Missing {name} in prml.dimreduce")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_sequential_exports(self):
        expected = ["GaussianHMM", "KalmanFilter"]
        for name in expected:
            self.assertTrue(hasattr(sequential, name), f"Missing {name} in prml.sequential")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_ensemble_exports(self):
        expected = ["DecisionStump", "AdaBoostClassifier", "MixtureOfLinearRegressions"]
        for name in expected:
            self.assertTrue(hasattr(ensemble, name), f"Missing {name} in prml.ensemble")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_distributions_exports(self):
        expected = [
            "Gaussian1D", "MultivariateGaussian", "BetaDistribution", "DirichletDistribution",
            "GammaDistribution", "StudentsTDistribution", "VonMisesDistribution",
            "KernelDensityEstimator", "KNearestNeighborsDensity", "RobbinsMonro"
        ]
        for name in expected:
            self.assertTrue(hasattr(distributions, name), f"Missing {name} in prml.distributions")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_graphical_exports(self):
        expected = [
            "SimpleFactorGraphChain", "check_d_separation", "denoise_image_icm",
            "noisy_or", "linear_gaussian_moments"
        ]
        for name in expected:
            self.assertTrue(hasattr(graphical, name), f"Missing {name} in prml.graphical")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_clustering_exports(self):
        expected = [
            "KMeans", "GaussianMixtureModel", "BernoulliMixtureModel",
            "VariationalGaussianMixture", "mixture_moments", "incremental_em_update"
        ]
        for name in expected:
            self.assertTrue(hasattr(clustering, name), f"Missing {name} in prml.clustering")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")

    def test_variational_exports(self):
        expected = [
            "VariationalGaussian1D", "VariationalGaussianMixture",
            "jaakkola_jordan_lambda", "ep_clutter_step", "variational_linear_regression"
        ]
        for name in expected:
            self.assertTrue(hasattr(variational, name), f"Missing {name} in prml.variational")
            self.assertTrue(hasattr(prml, name), f"Missing {name} in prml top-level")



    def test_fit_predict_interface_consistency(self):
        # 推定器が共通の fit / predict インターフェースを持つことを検証
        estimators = [
            linear.LinearRegression(),
            linear.RidgeRegression(),
            kernel.KernelRidgeRegression(),
            dimreduce.PCA(),
            clustering.KMeans(),
        ]
        for est in estimators:
            self.assertTrue(callable(getattr(est, 'fit', None)), f"{est} must have fit()")
            # PCA 以外は predict を持つ
            if not isinstance(est, dimreduce.PCA):
                self.assertTrue(callable(getattr(est, 'predict', None)), f"{est} must have predict()")

if __name__ == '__main__':
    unittest.main()
