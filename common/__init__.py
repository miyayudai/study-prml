"""
PRML (Pattern Recognition and Machine Learning) - Common Utilities
Bishopの理論とアルゴリズムを忠実にスクラッチ実装した共通機械学習ライブラリ
"""

from .plot_utils import setup_style, save_plot
from .distribution_utils import simplex_to_xy, plot_dirichlet_contour
from .regression_utils import (
    PolynomialBasis, GaussianBasis, SigmoidalBasis, FourierBasis,
    LinearRegression, RidgeRegression, LassoRegression, LeastMeanSquares,
    SequentialLinearRegression, WeightedLinearRegression, MultivariateLinearRegression,
    BayesianLinearRegression, NormalGammaLinearRegression, EvidenceApproximation,
    LocallyWeightedRegression, RobustLinearRegression, EquivalentKernel,
    bias_variance_decomposition, orthogonal_projection_matrix, equivalent_kernel_matrix,
    bayesian_model_evidence
)
from .classification_utils import (
    Perceptron, FisherLinearDiscriminant, MulticlassFisherLinearDiscriminant,
    GaussianGenerativeClassifier, LogisticRegression, MulticlassLogisticRegression,
    ProbitRegression, BayesianLogisticRegression, LaplaceApproximation
)
from .nn_utils import (MLPRegressor, MixtureDensityNetwork)
from .kernel_utils import (
    KernelRidgeRegression, NadarayaWatsonRegressor,
    GaussianProcessRegressor, GaussianProcessClassifier
)
from .svm_rvm_utils import SupportVectorClassifier, RelevanceVectorRegressor, RelevanceVectorClassifier
from .graphical_models_utils import SimpleFactorGraphChain
from .mixture_em_utils import KMeans, GaussianMixtureModel, BernoulliMixtureModel
from .variational_utils import VariationalGaussianMixture
from .sampling_utils import rejection_sample, metropolis_hastings, gibbs_sampler_2d, hamiltonian_monte_carlo
from .pca_ppca_utils import PCA, ProbabilisticPCA, KernelPCA
from .sequential_utils import GaussianHMM, KalmanFilter
from .ensemble_utils import DecisionStump, AdaBoostClassifier, MixtureOfLinearRegressions

__all__ = [
    # Plotting
    'setup_style', 'save_plot',
    # Ch 2: Distributions
    'simplex_to_xy', 'plot_dirichlet_contour',
    # Ch 3: Regression
    'PolynomialBasis', 'GaussianBasis', 'SigmoidalBasis', 'FourierBasis',
    'LinearRegression', 'RidgeRegression', 'LassoRegression', 'LeastMeanSquares',
    'SequentialLinearRegression', 'WeightedLinearRegression', 'MultivariateLinearRegression',
    'BayesianLinearRegression', 'NormalGammaLinearRegression', 'EvidenceApproximation',
    'LocallyWeightedRegression', 'RobustLinearRegression', 'EquivalentKernel',
    'bias_variance_decomposition', 'orthogonal_projection_matrix', 'equivalent_kernel_matrix',
    'bayesian_model_evidence',
    # Ch 4: Classification
    'Perceptron', 'FisherLinearDiscriminant', 'MulticlassFisherLinearDiscriminant',
    'GaussianGenerativeClassifier', 'LogisticRegression', 'MulticlassLogisticRegression',
    'ProbitRegression', 'BayesianLogisticRegression', 'LaplaceApproximation',
    # Ch 5: Neural Networks
    'MLPRegressor', 'MixtureDensityNetwork',
    # Ch 6: Kernel Methods
    'KernelRidgeRegression', 'NadarayaWatsonRegressor',
    'GaussianProcessRegressor', 'GaussianProcessClassifier',
    # Ch 7: Sparse Kernel Machines
    'SupportVectorClassifier', 'RelevanceVectorRegressor', 'RelevanceVectorClassifier',
    # Ch 8: Graphical Models
    'SimpleFactorGraphChain',
    # Ch 9: Mixture Models & EM
    'KMeans', 'GaussianMixtureModel', 'BernoulliMixtureModel',
    # Ch 10: Approximate Inference
    'VariationalGaussianMixture',
    # Ch 11: Sampling Methods
    'rejection_sample', 'metropolis_hastings', 'gibbs_sampler_2d', 'hamiltonian_monte_carlo',
    # Ch 12: Continuous Latent Variables
    'PCA', 'ProbabilisticPCA', 'KernelPCA',
    # Ch 13: Sequential Data
    'GaussianHMM', 'KalmanFilter',
    # Ch 14: Combining Models
    'DecisionStump', 'AdaBoostClassifier', 'MixtureOfLinearRegressions',
]
