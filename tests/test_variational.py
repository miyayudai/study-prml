import unittest
import numpy as np
import prml


class TestVariationalInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_variational_gaussian_1d_cavi(self):
        true_mu = 2.5
        true_sigma = 1.2
        true_tau = 1.0 / (true_sigma ** 2)
        X = np.random.normal(loc=true_mu, scale=true_sigma, size=300)
        
        cavi = prml.VariationalGaussian1D(max_iter=30)
        cavi.fit(X)
        
        self.assertIsNotNone(cavi.mean_mu)
        self.assertIsNotNone(cavi.mean_tau)
        self.assertAlmostEqual(cavi.mean_mu, np.mean(X), places=3)
        self.assertGreater(cavi.mean_tau, 0.0)
        self.assertAlmostEqual(cavi.mean_tau, 1.0 / np.var(X), delta=0.1)

    def test_jaakkola_jordan_bound(self):
        # xi = 0 should give 1/8 = 0.125
        val_0 = prml.jaakkola_jordan_lambda(0.0)
        self.assertAlmostEqual(float(val_0), 0.125)
        
        # Symmetry: lambda(xi) == lambda(-xi)
        xis = np.array([0.5, 1.0, 2.5, -1.0])
        lambdas = prml.jaakkola_jordan_lambda(xis)
        self.assertAlmostEqual(float(lambdas[1]), float(lambdas[3]))


if __name__ == '__main__':
    unittest.main()
