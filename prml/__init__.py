"""
PRML (Pattern Recognition and Machine Learning) Python Package
Christopher M. Bishop の名著に基づくスクラッチ機械学習実装ライブラリ

サブパッケージ構成:
- prml.linear: 線形回帰および線形分類モデル (Ch 3, 4)
- prml.kernel: カーネル法、ガウス過程、SVM、RVM (Ch 6, 7)
- prml.nn: ニューラルネットワーク、MDN (Ch 5)
- prml.clustering: K-means、GMM、EMアルゴリズム、変分推論 (Ch 9, 10)
- prml.sampling: モンテカルロ法、MCMC、HMC (Ch 11)
- prml.dimreduce: PCA、確率的PCA、カーネルPCA (Ch 12)
- prml.sequential: HMM、カルマンフィルタ (Ch 13)
- prml.ensemble: 決定木、AdaBoost、混合エキスパート (Ch 14)
- prml.graphical: 因子グラフ、確率伝播、d分離性 (Ch 8)
- prml.distributions: 基礎確率分布、ディリクレ単体変換 (Ch 2)
"""

from . import linear
from . import kernel
from . import nn
from . import clustering
from . import sampling
from . import dimreduce
from . import sequential
from . import ensemble
from . import graphical
from . import distributions

# Top-level direct access to core models & algorithms
from .linear import (
    LinearRegression,
    RidgeRegression,
    BayesianLinearRegression,
    EvidenceApproximation,
    PolynomialBasis,
    GaussianBasis,
    SigmoidalBasis,
    Perceptron,
    FisherLinearDiscriminant,
    GaussianGenerativeClassifier,
    LogisticRegression,
    MulticlassLogisticRegression,
    BayesianLogisticRegression,
)

from .kernel import (
    KernelRidgeRegression,
    NadarayaWatsonRegressor,
    GaussianProcessRegressor,
    GaussianProcessClassifier,
    SupportVectorClassifier,
    RelevanceVectorRegressor,
    RelevanceVectorClassifier,
)

from .nn import (
    MLPRegressor,
    MixtureDensityNetwork,
)

from .clustering import (
    KMeans,
    GaussianMixtureModel,
    BernoulliMixtureModel,
    VariationalGaussianMixture,
)

from .sampling import (
    rejection_sample,
    metropolis_hastings,
    gibbs_sampler_2d,
    hamiltonian_monte_carlo,
)

from .dimreduce import (
    PCA,
    ProbabilisticPCA,
    KernelPCA,
)

from .sequential import (
    GaussianHMM,
    KalmanFilter,
)

from .ensemble import (
    DecisionStump,
    AdaBoostClassifier,
    MixtureOfLinearRegressions,
)

from .graphical import (
    SimpleFactorGraphChain,
    check_d_separation,
)

from .distributions import (
    simplex_to_xy,
    plot_dirichlet_contour,
)

__version__ = "1.1.0"
__author__ = "miyayudai"

__all__ = [
    # Submodules
    "linear",
    "kernel",
    "nn",
    "clustering",
    "sampling",
    "dimreduce",
    "sequential",
    "ensemble",
    "graphical",
    "distributions",
    # Core Models
    "LinearRegression",
    "RidgeRegression",
    "BayesianLinearRegression",
    "EvidenceApproximation",
    "PolynomialBasis",
    "GaussianBasis",
    "SigmoidalBasis",
    "Perceptron",
    "FisherLinearDiscriminant",
    "GaussianGenerativeClassifier",
    "LogisticRegression",
    "MulticlassLogisticRegression",
    "BayesianLogisticRegression",
    "KernelRidgeRegression",
    "NadarayaWatsonRegressor",
    "GaussianProcessRegressor",
    "GaussianProcessClassifier",
    "SupportVectorClassifier",
    "RelevanceVectorRegressor",
    "RelevanceVectorClassifier",
    "MLPRegressor",
    "MixtureDensityNetwork",
    "KMeans",
    "GaussianMixtureModel",
    "BernoulliMixtureModel",
    "VariationalGaussianMixture",
    "rejection_sample",
    "metropolis_hastings",
    "gibbs_sampler_2d",
    "hamiltonian_monte_carlo",
    "PCA",
    "ProbabilisticPCA",
    "KernelPCA",
    "GaussianHMM",
    "KalmanFilter",
    "DecisionStump",
    "AdaBoostClassifier",
    "MixtureOfLinearRegressions",
    "SimpleFactorGraphChain",
    "check_d_separation",
    "simplex_to_xy",
    "plot_dirichlet_contour",
]
