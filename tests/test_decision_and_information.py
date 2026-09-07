import unittest
import numpy as np
import prml


class TestDecisionTheory(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_bayes_decision_classifier_zero_one_loss(self):
        # 3 classes, posteriors
        probas = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        clf = prml.BayesDecisionClassifier()
        clf.fit_posteriors(probas)
        preds = clf.predict(probas)
        np.testing.assert_array_equal(preds, [0, 1, 2])

    def test_bayes_decision_asymmetric_loss(self):
        # 2 classes: 0 = normal, 1 = cancer
        # False negative (classifying 1 as 0) penalty = 50, False positive = 1
        loss_matrix = np.array([
            [0.0, 1.0],   # true is 0: pred 0 cost 0, pred 1 cost 1
            [50.0, 0.0]   # true is 1: pred 0 cost 50, pred 1 cost 0
        ])
        clf = prml.BayesDecisionClassifier(loss_matrix=loss_matrix)
        # Even with low cancer posterior (e.g. 0.10), high penalty makes it predict cancer (1)
        probas = np.array([
            [0.90, 0.10],  # expected loss for 0: 0.1*50 = 5.0, for 1: 0.9*1 = 0.9 -> predict 1
            [0.99, 0.01]   # expected loss for 0: 0.01*50 = 0.5, for 1: 0.99*1 = 0.99 -> predict 0
        ])
        preds = clf.predict(probas)
        self.assertEqual(preds[0], 1)
        self.assertEqual(preds[1], 0)

    def test_reject_option(self):
        probas = np.array([
            [0.9, 0.1],
            [0.55, 0.45],
            [0.48, 0.52]
        ])
        reject_clf = prml.RejectOptionClassifier(theta=0.7, reject_label=-1)
        preds = reject_clf.predict(probas)
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], -1)  # max prob 0.55 < 0.7
        self.assertEqual(preds[2], -1)  # max prob 0.52 < 0.7

    def test_reject_tradeoff_evaluation(self):
        y_true = np.array([0, 0, 1, 1])
        probas = np.array([
            [0.9, 0.1],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.2, 0.8]
        ])
        reject_clf = prml.RejectOptionClassifier()
        thetas, reject_rates, error_rates = reject_clf.evaluate_tradeoff(y_true, probas, thetas=[0.5, 0.7, 0.95])
        self.assertEqual(len(thetas), 3)
        self.assertLessEqual(reject_rates[0], reject_rates[-1])

    def test_minkowski_loss(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 1.0])
        # diffs: 1, 0, -2 -> abs: 1, 0, 2
        # q=1: mean(1, 0, 2) = 1.0
        self.assertAlmostEqual(prml.minkowski_loss(y_true, y_pred, q=1.0), 1.0)
        # q=2: mean(1, 0, 4) = 5/3
        self.assertAlmostEqual(prml.minkowski_loss(y_true, y_pred, q=2.0), 5.0 / 3.0)

    def test_roc_curve(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, thresholds, auc = prml.compute_roc_curve(y_true, y_score)
        self.assertGreaterEqual(auc, 0.5)
        self.assertLessEqual(auc, 1.0)
        self.assertEqual(fpr[0], 0.0)
        self.assertEqual(tpr[0], 0.0)
        self.assertEqual(fpr[-1], 1.0)
        self.assertEqual(tpr[-1], 1.0)


class TestInformationTheory(unittest.TestCase):
    def test_entropy_discrete(self):
        # Fair coin: 1.0 bit
        h_fair = prml.entropy_discrete([0.5, 0.5], base=2.0)
        self.assertAlmostEqual(h_fair, 1.0)

        # Deterministic: 0.0
        h_det = prml.entropy_discrete([1.0, 0.0], base=2.0)
        self.assertAlmostEqual(h_det, 0.0)

        # Uniform 4 outcomes: log2(4) = 2.0 bits
        h_4 = prml.entropy_discrete([0.25, 0.25, 0.25, 0.25], base=2.0)
        self.assertAlmostEqual(h_4, 2.0)

    def test_binary_entropy(self):
        self.assertAlmostEqual(float(prml.binary_entropy(0.5)), 1.0)
        self.assertAlmostEqual(float(prml.binary_entropy(0.0)), 0.0, places=5)
        self.assertAlmostEqual(float(prml.binary_entropy(1.0)), 0.0, places=5)

    def test_differential_entropy_gaussian(self):
        # 1D with variance 1.0: 0.5 * ln(2 pi e)
        h_gauss_1d = prml.differential_entropy_gaussian(1.0)
        expected = 0.5 * np.log(2 * np.pi * np.e)
        self.assertAlmostEqual(h_gauss_1d, expected)

    def test_kl_divergence_discrete(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        self.assertAlmostEqual(prml.kl_divergence_discrete(p, q), 0.0)

        q2 = np.array([0.9, 0.1])
        kl = prml.kl_divergence_discrete(p, q2)
        self.assertGreater(kl, 0.0)

    def test_kl_divergence_gaussian(self):
        mu = np.array([0.0])
        cov = np.array([[1.0]])
        # Same distribution: KL = 0
        self.assertAlmostEqual(prml.kl_divergence_gaussian(mu, cov, mu, cov), 0.0)

        # Different mean
        mu2 = np.array([1.0])
        kl = prml.kl_divergence_gaussian(mu, cov, mu2, cov)
        self.assertAlmostEqual(kl, 0.5)

    def test_mutual_information_gaussian(self):
        # Independent: rho = 0 -> MI = 0
        self.assertAlmostEqual(float(prml.mutual_information_gaussian(0.0)), 0.0)
        
        # High correlation: rho = 0.8
        mi = float(prml.mutual_information_gaussian(0.8))
        expected = -0.5 * np.log(1.0 - 0.64)
        self.assertAlmostEqual(mi, expected)

    def test_huffman_coding(self):
        probs = {'A': 0.5, 'B': 0.25, 'C': 0.125, 'D': 0.125}
        codes, avg_len = prml.huffman_coding(probs)
        # Prefix free codes
        code_values = list(codes.values())
        for i, c1 in enumerate(code_values):
            for j, c2 in enumerate(code_values):
                if i != j:
                    self.assertFalse(c2.startswith(c1))
        # Average length equals Shannon entropy for power of two
        h = prml.entropy_discrete(list(probs.values()), base=2.0)
        self.assertAlmostEqual(avg_len, h)


if __name__ == '__main__':
    unittest.main()
