"""
PRML (Pattern Recognition and Machine Learning) Python Package
Christopher M. Bishop の名著に基づくスクラッチ機械学習実装ライブラリ

サブパッケージ構成:
- prml.linear: 線形回帰および線形分類モデル (Ch 3, 4)
- prml.decision: 決定理論、損失関数、ROC曲線、棄却オプション (Ch 1.5)
- prml.information: 情報理論、シャノンエントロピー、KLダイバージェンス、相互情報量 (Ch 1.6)
- prml.kernel: カーネル法、ガウス過程、SVM、RVM (Ch 6, 7)
- prml.nn: ニューラルネットワーク、MDN (Ch 5)
- prml.clustering: K-means、GMM、EMアルゴリズム (Ch 9)
- prml.variational: 変分ベイズ推論、CAVI、局所変分境界 (Ch 10)
- prml.sampling: モンテカルロ法、MCMC、HMC (Ch 11)
- prml.dimreduce: PCA、確率的PCA、カーネルPCA (Ch 12)
- prml.sequential: HMM、カルマンフィルタ (Ch 13)
- prml.ensemble: 決定木、AdaBoost、混合エキスパート (Ch 14)
- prml.graphical: 因子グラフ、確率伝播、d分離性 (Ch 8)
- prml.distributions: 基礎確率分布、ディリクレ単体変換 (Ch 2)
"""

from . import linear
from . import decision
from . import information
from . import kernel
from . import nn
from . import clustering
from . import variational
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
    LassoRegression,
    LeastMeanSquares,
    SequentialLinearRegression,
    WeightedLinearRegression,
    MultivariateLinearRegression,
    BayesianLinearRegression,
    NormalGammaLinearRegression,
    EvidenceApproximation,
    LocallyWeightedRegression,
    RobustLinearRegression,
    EquivalentKernel,
    PolynomialBasis,
    GaussianBasis,
    SigmoidalBasis,
    FourierBasis,
    bias_variance_decomposition,
    orthogonal_projection_matrix,
    equivalent_kernel_matrix,
    bayesian_model_evidence,
    Perceptron,
    FisherLinearDiscriminant,
    MulticlassFisherLinearDiscriminant,
    GaussianGenerativeClassifier,
    LogisticRegression,
    MulticlassLogisticRegression,
    ProbitRegression,
    BayesianLogisticRegression,
    LaplaceApproximation,
)

from .decision import (
    BayesDecisionClassifier,
    RejectOptionClassifier,
    minkowski_loss,
    compute_roc_curve,
)

from .information import (
    entropy_discrete,
    binary_entropy,
    differential_entropy_gaussian,
    kl_divergence_discrete,
    kl_divergence_gaussian,
    mutual_information_gaussian,
    huffman_coding,
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

from .variational import (
    VariationalGaussian1D,
    jaakkola_jordan_lambda,
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
    denoise_image_icm,
    noisy_or,
    linear_gaussian_moments,
)

from .distributions import (
    Gaussian1D,
    MultivariateGaussian,
    BetaDistribution,
    DirichletDistribution,
    GammaDistribution,
    StudentsTDistribution,
    VonMisesDistribution,
    KernelDensityEstimator,
    KNearestNeighborsDensity,
    RobbinsMonro,
    simplex_to_xy,
    plot_dirichlet_contour,
    plot_gaussian_ellipse,
    student_t_pdf,
    von_mises_pdf,
)

__version__ = "1.3.0"
__author__ = "miyayudai"

__all__ = [
    # Submodules
    "linear",
    "decision",
    "information",
    "kernel",
    "nn",
    "clustering",
    "variational",
    "sampling",
    "dimreduce",
    "sequential",
    "ensemble",
    "graphical",
    "distributions",
    # Core Models & Functions
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "LeastMeanSquares",
    "SequentialLinearRegression",
    "WeightedLinearRegression",
    "MultivariateLinearRegression",
    "BayesianLinearRegression",
    "NormalGammaLinearRegression",
    "EvidenceApproximation",
    "LocallyWeightedRegression",
    "RobustLinearRegression",
    "EquivalentKernel",
    "PolynomialBasis",
    "GaussianBasis",
    "SigmoidalBasis",
    "FourierBasis",
    "bias_variance_decomposition",
    "orthogonal_projection_matrix",
    "equivalent_kernel_matrix",
    "bayesian_model_evidence",
    "Perceptron",
    "FisherLinearDiscriminant",
    "MulticlassFisherLinearDiscriminant",
    "GaussianGenerativeClassifier",
    "LogisticRegression",
    "MulticlassLogisticRegression",
    "ProbitRegression",
    "BayesianLogisticRegression",
    "LaplaceApproximation",
    "BayesDecisionClassifier",
    "RejectOptionClassifier",
    "minkowski_loss",
    "compute_roc_curve",
    "entropy_discrete",
    "binary_entropy",
    "differential_entropy_gaussian",
    "kl_divergence_discrete",
    "kl_divergence_gaussian",
    "mutual_information_gaussian",
    "huffman_coding",
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
    "VariationalGaussian1D",
    "jaakkola_jordan_lambda",
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
    "denoise_image_icm",
    "noisy_or",
    "linear_gaussian_moments",
    "Gaussian1D",
    "MultivariateGaussian",
    "BetaDistribution",
    "DirichletDistribution",
    "GammaDistribution",
    "StudentsTDistribution",
    "VonMisesDistribution",
    "KernelDensityEstimator",
    "KNearestNeighborsDensity",
    "RobbinsMonro",
    "simplex_to_xy",
    "plot_dirichlet_contour",
    "plot_gaussian_ellipse",
    "student_t_pdf",
    "von_mises_pdf",
]
