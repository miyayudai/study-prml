"""
PRML Linear Models Module (Chapters 3 & 4)
線形回帰および線形分類モデル
- 基底関数: PolynomialBasis, GaussianBasis, SigmoidalBasis, FourierBasis
- 線形回帰: LinearRegression, RidgeRegression, LassoRegression, LeastMeanSquares (LMS),
           SequentialLinearRegression, WeightedLinearRegression, MultivariateLinearRegression,
           LocallyWeightedRegression, RobustLinearRegression
- ベイズ線形回帰: BayesianLinearRegression, NormalGammaLinearRegression, EvidenceApproximation,
                  EquivalentKernel, bayesian_model_evidence, orthogonal_projection_matrix,
                  equivalent_kernel_matrix, BiasVarianceSimulator, bias_variance_decomposition
- 線形分類: Perceptron, FisherLinearDiscriminant, MulticlassFisherLinearDiscriminant,
           GaussianGenerativeClassifier, LogisticRegression, MulticlassLogisticRegression,
           ProbitRegression, BayesianLogisticRegression, LaplaceApproximation
"""

from common.regression_utils import (
    PolynomialBasis,
    GaussianBasis,
    SigmoidalBasis,
    FourierBasis,
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LeastMeanSquares,
    LMS,
    SequentialLinearRegression,
    WeightedLinearRegression,
    MultivariateLinearRegression,
    BayesianLinearRegression,
    NormalGammaLinearRegression,
    EvidenceApproximation,
    LocallyWeightedRegression,
    RobustLinearRegression,
    EquivalentKernel,
    BiasVarianceSimulator,
    bias_variance_decomposition,
    orthogonal_projection_matrix,
    equivalent_kernel_matrix,
    bayesian_model_evidence,
)

from common.classification_utils import (
    sigmoid,
    softmax,
    Perceptron,
    FisherLinearDiscriminant,
    MulticlassFisherLinearDiscriminant,
    GaussianGenerativeClassifier,
    LogisticRegression,
    MulticlassLogisticRegression,
    ProbitRegression,
    BayesianLogisticRegression,
    LaplaceApproximation,
    plot_decision_boundary_2d,
)

__all__ = [
    # Basis functions
    "PolynomialBasis",
    "GaussianBasis",
    "SigmoidalBasis",
    "FourierBasis",
    # Linear Regression Models
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "LeastMeanSquares",
    "LMS",
    "SequentialLinearRegression",
    "WeightedLinearRegression",
    "MultivariateLinearRegression",
    "LocallyWeightedRegression",
    "RobustLinearRegression",
    # Bayesian Linear Regression & Model Selection
    "BayesianLinearRegression",
    "NormalGammaLinearRegression",
    "EvidenceApproximation",
    "EquivalentKernel",
    "BiasVarianceSimulator",
    "bias_variance_decomposition",
    "orthogonal_projection_matrix",
    "equivalent_kernel_matrix",
    "bayesian_model_evidence",
    # Linear Classification Models
    "sigmoid",
    "softmax",
    "Perceptron",
    "FisherLinearDiscriminant",
    "MulticlassFisherLinearDiscriminant",
    "GaussianGenerativeClassifier",
    "LogisticRegression",
    "MulticlassLogisticRegression",
    "ProbitRegression",
    "BayesianLogisticRegression",
    "LaplaceApproximation",
    "plot_decision_boundary_2d",
]
