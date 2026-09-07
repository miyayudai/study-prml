"""
PRML Chapter 10: Approximate Inference & Variational Inference
変分ベイズ推論モデル
- 1変量ガウス分布の変分推論 (VariationalGaussian1D, CAVI)
- 変分混合ガウスモデル (VariationalGaussianMixture)
- 局所変分境界関数 (jaakkola_jordan_bound, jaakkola_jordan_lambda)
"""

import numpy as np
from common.variational_utils import (
    variational_gaussian_1d,
    VariationalGaussianMixture,
    jaakkola_jordan_lambda,
)


class VariationalGaussian1D:
    """
    PRML 10.1.3: 1変量ガウス分布の平均 mu と精度 tau に対する座標上昇変分推論 (CAVI)
    
    モデル:
        p(D | mu, tau) = \\prod_{n=1}^N N(x_n | mu, tau^{-1})
    事前分布:
        p(mu | tau) = N(mu | mu_0, (lambda_0 tau)^{-1})
        p(tau) = Gamma(tau | a_0, b_0)
    平均場近似:
        q(mu, tau) = q_mu(mu) q_tau(tau)
        q_mu(mu) = N(mu | mu_N, lambda_N^{-1})
        q_tau(tau) = Gamma(tau | a_N, b_N)
    """

    def __init__(self, mu_0=0.0, lambda_0=0.0, a_0=0.0, b_0=0.0, max_iter=25):
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.a_0 = a_0
        self.b_0 = b_0
        self.max_iter = max_iter
        
        self.mu_N_ = None
        self.lambda_N_ = None
        self.a_N_ = None
        self.b_N_ = None
        self.history_ = []

    def fit(self, X):
        X = np.asarray(X).ravel()
        history = variational_gaussian_1d(
            X,
            mu_0=self.mu_0,
            lambda_0=self.lambda_0,
            a_0=self.a_0,
            b_0=self.b_0,
            max_iter=self.max_iter,
        )
        self.history_ = history
        if len(history) > 0:
            self.mu_N_, self.lambda_N_, self.a_N_, self.b_N_ = history[-1]
        return self

    @property
    def mean_mu(self):
        return self.mu_N_

    @property
    def mean_tau(self):
        return self.a_N_ / self.b_N_


__all__ = [
    "VariationalGaussian1D",
    "VariationalGaussianMixture",
    "jaakkola_jordan_lambda",
]
