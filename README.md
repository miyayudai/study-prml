# PRML (Pattern Recognition and Machine Learning) Python Implementation

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-255%20passed-success.svg)](tests/)
[![PRML](https://img.shields.io/badge/PRML-Complete%20All%20Chapters-brightgreen.svg)](TASK.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Christopher M. Bishop の世界的名著『**Pattern Recognition and Machine Learning (PRML)**』の全14章＋第0章（確率論の基礎補講）を、理論解説・数式展開・Pythonスクラッチ実装・可視化シミュレーション・全章演習問題（穴埋め＆証明解説）として完全実装したリポジトリです。

---

## 🌟 主な特徴

1. **第0章〜第14章（全15章・全64冊）の完全Notebook化**
   - 原著の全節（1.1〜1.6、2.1〜2.5...）を網羅した詳細なJupyter Notebookと、113枚を超える高精細な再現図版。
   - 理論（数式導出・LaTeX）と実装（Pythonスクラッチコード）を一体化。
2. **全章の演習問題（Exercises・全370問超）を完全網羅**
   - 穴埋め形式（`None` / `# YOUR CODE HERE` / 語句穴埋め `___`）および厳密なPython数値検証コードを完備。独学や演習ゼミに最適。
3. **高品質な `prml` パッケージとしてのライブラリ化**
   - `pip install -e .` でインストール可能。PRMLで登場する全ての基盤アルゴリズムを統一されたモダンなAPIで提供。
4. **包括的テストスイート (255 Tests / 128 Subtests)**
   - **255件** の単体・統合テストおよび全64冊ノートブック完全実行・エラーゼロ自動整合性検証テストにより、数理的整合性・境界条件・収束性・規格適合性を厳密に検証（約5秒で全件通過）。CI による継続的自動テストも整備。


---

## 📂 ディレクトリ構成

```text
my_PRML/
├── 0/               # 第0章: 確率論の基礎 (加法・乗法・ベイズの定理・変数変換)
├── 1/               # 第1章: 序論 (曲線あてはめ・確率論・決定理論・情報理論・演習問題全41問)
├── 2/               # 第2章: 確率分布 (二値・多項・ガウス・指数型分布族・ノンパラメトリック)
├── 3/               # 第3章: 線形回帰モデル (最尤推定・正則化・ベイズ線形回帰・エビデンス)
├── 4/               # 第4章: 線形分類モデル (判別関数・生成モデル・ロジスティック回帰・ラプラス近似)
├── 5/               # 第5章: ニューラルネットワーク (誤差逆伝播・正則化・混合密度ネットワーク)
├── 6/               # 第6章: カーネル法 (双対表現・RBF・Nadaraya-Watson・ガウス過程回帰/分類)
├── 7/               # 第7章: スパースカーネルマシン (SVM・SMO・関連ベクトルマシン RVM)
├── 8/               # 第8章: グラフィカルモデル (ベイジアンネット・マルコフ無向グラフ・Sum-Product)
├── 9/               # 第9章: 混合モデルとEMアルゴリズム (K-Means・GMM・ベルヌーイ混合・一般化EM)
├── 10/              # 第10章: 近似推論法 (変分ベイズ・変分混合ガウス VB-GMM・局所変分法)
├── 11/              # 第11章: サンプリング法 (棄却サンプリング・M-H法・ギブスサンプリング・HMC)
├── 12/              # 第12章: 連続潜在変数 (主成分分析 PCA・確率的PCA PPCA・カーネルPCA)
├── 13/              # 第13章: 系列データ (隠れマルコフモデル HMM・カルマンフィルタ LDS)
├── 14/              # 第14章: モデルの結合 (バギング・ブースティング AdaBoost・線形回帰混合)
├── common/          # スクラッチ機械学習基盤ライブラリ (アルゴリズム・可視化・データ)
├── prml/            # 体系化された PRML Python パッケージ
│   ├── linear.py         # 線形回帰・分類 (Ch 3, 4)
│   ├── kernel.py         # カーネル法・SVM・RVM (Ch 6, 7)
│   ├── nn.py             # ニューラルネットワーク・MDN (Ch 5)
│   ├── clustering.py     # クラスタリング・EM・変分混合 (Ch 9, 10)
│   ├── sampling.py       # MCMC・HMCサンプリング (Ch 11)
│   ├── dimreduce.py      # 主成分分析・潜在変数 (Ch 12)
│   ├── sequential.py     # HMM・カルマンフィルタ (Ch 13)
│   ├── ensemble.py       # AdaBoost・混合エキスパート (Ch 14)
│   ├── graphical.py      # 因子グラフ・d分離 (Ch 8)
│   └── distributions.py  # 基礎分布・単体変換 (Ch 2)
├── scripts/         # ノートブック自動生成・検証用スクリプト群 (50+ scripts)
├── tests/           # 統合・単体テストスイート (255 tests / 128 subtests)
├── pyproject.toml   # PEP 517/621 パッケージ定義ファイル
├── setup.py         # セットアップスクリプト
├── TASK.md          # 開発要件・進捗管理ドキュメント (100% 完了)
└── README.md        # 本ドキュメント
```

---

## 🚀 クイックスタート

### 1. リポジトリのクローン & パッケージインストール

```bash
git clone https://github.com/miyayudai/my_PRML.git
cd my_PRML

# 開発モードでインストール (prml パッケージが利用可能になります)
pip install -e .
```

### 2. ライブラリとしての利用例

Bishop のアルゴリズムはすべて `scikit-learn` に近い直感的なインターフェースで設計されています：

```python
import numpy as np

# サブパッケージからの個別インポート、またはトップレベルからの直接インポートに対応
from prml.linear import BayesianLinearRegression
from prml.kernel import GaussianProcessRegressor
from prml.clustering import GaussianMixtureModel

# 1. ベイズ線形回帰 (Ch 3)
X = np.random.randn(50, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + np.random.normal(0, 0.1, 50)

model = BayesianLinearRegression(alpha=1.0, beta=100.0)
model.fit(X, y)
mean, var = model.predict(X)
print(f"予測平均: {mean.shape}, 予測分散: {var.shape}")

# 2. ガウス過程回帰 (Ch 6)
gpr = GaussianProcessRegressor(kernel='rbf', gamma=1.0, beta=50.0)
gpr.fit(X, y)
gpr_mean, gpr_var = gpr.predict(X)

# 3. 混合ガウスモデル (EMアルゴリズム, Ch 9)
gmm = GaussianMixtureModel(n_components=2)
gmm.fit(X)
labels = gmm.predict(X)
```

### 3. ユニットテストの実行

```bash
# 全15章・全モジュール・全ノートブック完全実行自動検査を含む包括的テスト (255 tests, 約5秒で全件通過)
pytest tests -v
```

---

## 📚 章別カリキュラムと主要トピック

| 章 | タイトル | 主なトピック・実装アルゴリズム | 演習問題 (Exercises) | 再現図版 |
|:---:|---|---|:---:|:---:|
| **0** | **確率論の基礎** | 加法定理、乗法定理、ベイズの定理、変数変換定理、Jensen不等式、KL情報量 | Ex 0.1〜0.6 (全6問) | 3 枚 |
| **1** | **序論** | 多項式曲線あてはめ、確率論基礎、モデル選択、決定理論、損失関数、情報理論 | Ex 1.1〜1.41 (全41問) | 20 枚 |
| **2** | **確率分布** | 二値・多項分布、Dirichlet、1次元・多次元ガウス分布、von Mises 分布、ノンパラメトリック | Ex 2.1〜2.61 (全61問) | 13 枚 |
| **3** | **線形回帰モデル** | 最小二乗法、正則化最小二乗、ベイズ線形回帰、エビデンス近似、バイアス-バリアンス分解 | Ex 3.1〜3.24 (全24問) | 15 枚 |
| **4** | **線形分類モデル** | パーセプトロン、Fisher LDA、ロジスティック回帰、多クラスロジスティック、IRLS、ラプラス近似 | Ex 4.1〜4.26 (全26問) | 12 枚 |
| **5** | **ニューラルネットワーク** | 多層パーセプトロン、誤差逆伝播法、数値勾配チェック、正則化、混合密度ネットワーク (MDN) | Ex 5.1〜5.41 (全41問) | 7 枚 |
| **6** | **カーネル法** | 双対表現、RBF/ARDカーネル、Nadaraya-Watson 回帰、ガウス過程回帰 (GPR)、ガウス過程分類 (GPC) | Ex 6.1〜6.27 (全27問) | 8 枚 |
| **7** | **スパースカーネルマシン** | サポートベクトルマシン (SVC)、SMO、関連ベクトルマシン (RVM 回帰・分類) | Ex 7.1〜7.19 (全19問) | 6 枚 |
| **8** | **グラフィカルモデル** | ベイジアンネットワーク、d-分離、マルコフ確率場、Isingモデル、Factor Graph、Sum-Product法 | Ex 8.1〜8.29 (全29問) | 3 枚 |
| **9** | **混合モデルとEM** | K-Means、混合ガウスモデル (GMM)、ベルヌーイ混合モデル、潜在変数とEMアルゴリズム | Ex 9.1〜9.27 (全27問) | 6 枚 |
| **10** | **近似推論法** | 変分推論法 (Variational Inference)、変分混合ガウス (VB-GMM)、局所変分法 | Ex 10.1〜10.39 (全39問) | 5 枚 |
| **11** | **サンプリング法** | 棄却サンプリング、重要度サンプリング、M-Hアルゴリズム、Gibbsサンプリング、HMC | Ex 11.1〜11.17 (全17問) | 6 枚 |
| **12** | **連続潜在変数** | 主成分分析 (PCA)、確率的主成分分析 (PPCA)、EMアルゴリズムによるPPCA、Kernel PCA | Ex 12.1〜12.29 (全29問) | 5 枚 |
| **13** | **系列データ** | マルコフモデル、隠れマルコフモデル (HMM, Forward-Backward, Viterbi)、カルマンフィルタ (LDS) | Ex 13.1〜13.34 (全34問) | 4 枚 |
| **14** | **モデルの結合** | バギング、ブースティング (AdaBoost)、決定株 (Decision Stump)、局所エキスパート混合 (MoE) | Ex 14.1〜14.17 (全17問) | 5 枚 |

**合計: 15章・全64冊ノートブック・全370問超の演習問題完全網羅・118 枚の再現図版を収録！**

---


## 🔗 ノートブック一覧 (Colabで1クリック起動)

以下のバッジをクリックすると、各ノートブックをGoogle Colab上で直接開いて実行できます。先頭のセットアップセルを実行するだけで即座に学習を開始できます。

### 第0章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/0/0_Exercises.ipynb) `0_Exercises.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/0/0_Foundations_of_Probability.ipynb) `0_Foundations_of_Probability.ipynb`

### 第1章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/1/1.1_Polynomial_Curve_Fitting.ipynb) `1.1_Polynomial_Curve_Fitting.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/1/1.2_Probability_Theory.ipynb) `1.2_Probability_Theory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/1/1.3-1.4_Model_Selection_and_Curse_of_Dimensionality.ipynb) `1.3-1.4_Model_Selection_and_Curse_of_Dimensionality.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/1/1.5_Decision_Theory.ipynb) `1.5_Decision_Theory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/1/1.6_Information_Theory.ipynb) `1.6_Information_Theory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/1/1_Exercises.ipynb) `1_Exercises.ipynb`

### 第2章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/2/2.1_Binary_Variables.ipynb) `2.1_Binary_Variables.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/2/2.2_Multinomial_Variables.ipynb) `2.2_Multinomial_Variables.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/2/2.3_The_Gaussian_Distribution.ipynb) `2.3_The_Gaussian_Distribution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/2/2.4_The_Exponential_Family.ipynb) `2.4_The_Exponential_Family.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/2/2.5_Nonparametric_Methods.ipynb) `2.5_Nonparametric_Methods.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/2/2_Exercises.ipynb) `2_Exercises.ipynb`

### 第3章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/3/3.1_Linear_Basis_Function_Models.ipynb) `3.1_Linear_Basis_Function_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/3/3.2_The_Bias_Variance_Decomposition.ipynb) `3.2_The_Bias_Variance_Decomposition.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/3/3.3_Bayesian_Linear_Regression.ipynb) `3.3_Bayesian_Linear_Regression.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/3/3.4_Bayesian_Model_Comparison.ipynb) `3.4_Bayesian_Model_Comparison.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/3/3.5-3.6_The_Evidence_Approximation_and_Limitations.ipynb) `3.5-3.6_The_Evidence_Approximation_and_Limitations.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/3/3_Exercises.ipynb) `3_Exercises.ipynb`

### 第4章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/4/4.1_Discriminant_Functions.ipynb) `4.1_Discriminant_Functions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/4/4.2_Probabilistic_Generative_Models.ipynb) `4.2_Probabilistic_Generative_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/4/4.3_Probabilistic_Discriminative_Models.ipynb) `4.3_Probabilistic_Discriminative_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/4/4.4-4.5_Laplace_Approximation_and_Bayesian_Logistic_Regression.ipynb) `4.4-4.5_Laplace_Approximation_and_Bayesian_Logistic_Regression.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/4/4_Exercises.ipynb) `4_Exercises.ipynb`

### 第5章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/5/5.1-5.3_Feedforward_Networks_and_Backpropagation.ipynb) `5.1-5.3_Feedforward_Networks_and_Backpropagation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/5/5.4-5.5_The_Hessian_and_Regularization.ipynb) `5.4-5.5_The_Hessian_and_Regularization.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/5/5.6_Mixture_Density_Networks.ipynb) `5.6_Mixture_Density_Networks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/5/5.7_Bayesian_Neural_Networks.ipynb) `5.7_Bayesian_Neural_Networks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/5/5_Exercises.ipynb) `5_Exercises.ipynb`

### 第6章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/6/6.1-6.2_Dual_Representations_and_Kernel_Construction.ipynb) `6.1-6.2_Dual_Representations_and_Kernel_Construction.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/6/6.3_Radial_Basis_Function_Networks_and_Nadaraya_Watson.ipynb) `6.3_Radial_Basis_Function_Networks_and_Nadaraya_Watson.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/6/6.4_Gaussian_Processes_for_Regression_and_ARD.ipynb) `6.4_Gaussian_Processes_for_Regression_and_ARD.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/6/6.5_Gaussian_Processes_for_Classification.ipynb) `6.5_Gaussian_Processes_for_Classification.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/6/6_Exercises.ipynb) `6_Exercises.ipynb`

### 第7章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/7/7.1_Support_Vector_Machines.ipynb) `7.1_Support_Vector_Machines.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/7/7.2_Relevance_Vector_Machines.ipynb) `7.2_Relevance_Vector_Machines.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/7/7_Exercises.ipynb) `7_Exercises.ipynb`

### 第8章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/8/8.1-8.2_Bayesian_Networks_and_Conditional_Independence.ipynb) `8.1-8.2_Bayesian_Networks_and_Conditional_Independence.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/8/8.3_Markov_Random_Fields_and_Image_Denoising.ipynb) `8.3_Markov_Random_Fields_and_Image_Denoising.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/8/8.4_Exact_Inference_and_Factor_Graphs.ipynb) `8.4_Exact_Inference_and_Factor_Graphs.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/8/8_Exercises.ipynb) `8_Exercises.ipynb`

### 第9章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/9/9.1_K_means_Clustering.ipynb) `9.1_K_means_Clustering.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/9/9.2-9.3_Gaussian_and_Bernoulli_Mixture_Models.ipynb) `9.2-9.3_Gaussian_and_Bernoulli_Mixture_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/9/9.4_The_General_EM_Algorithm.ipynb) `9.4_The_General_EM_Algorithm.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/9/9_Exercises.ipynb) `9_Exercises.ipynb`

### 第10章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/10/10.1_Variational_Inference_Foundations.ipynb) `10.1_Variational_Inference_Foundations.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/10/10.2_Variational_Gaussian_Mixtures.ipynb) `10.2_Variational_Gaussian_Mixtures.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/10/10.5-10.7_Local_Variational_and_Expectation_Propagation.ipynb) `10.5-10.7_Local_Variational_and_Expectation_Propagation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/10/10_Exercises.ipynb) `10_Exercises.ipynb`

### 第11章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/11/11.1_Basic_Sampling_Algorithms.ipynb) `11.1_Basic_Sampling_Algorithms.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/11/11.2-11.3_MCMC_and_Gibbs_Sampling.ipynb) `11.2-11.3_MCMC_and_Gibbs_Sampling.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/11/11.5_Hybrid_Monte_Carlo.ipynb) `11.5_Hybrid_Monte_Carlo.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/11/11_Exercises.ipynb) `11_Exercises.ipynb`

### 第12章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/12/12.1_Principal_Component_Analysis.ipynb) `12.1_Principal_Component_Analysis.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/12/12.2_Probabilistic_PCA_and_Factor_Analysis.ipynb) `12.2_Probabilistic_PCA_and_Factor_Analysis.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/12/12.3_Kernel_PCA.ipynb) `12.3_Kernel_PCA.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/12/12_Exercises.ipynb) `12_Exercises.ipynb`

### 第13章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/13/13.1-13.2_Hidden_Markov_Models.ipynb) `13.1-13.2_Hidden_Markov_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/13/13.3_Linear_Dynamical_Systems_Kalman_Filter.ipynb) `13.3_Linear_Dynamical_Systems_Kalman_Filter.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/13/13_Exercises.ipynb) `13_Exercises.ipynb`

### 第14章
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/14/14.1-14.3_Bagging_and_AdaBoost.ipynb) `14.1-14.3_Bagging_and_AdaBoost.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/14/14.4-14.5_Decision_Trees_and_Mixture_of_Experts.ipynb) `14.4-14.5_Decision_Trees_and_Mixture_of_Experts.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_PRML/blob/main/14/14_Exercises.ipynb) `14_Exercises.ipynb`


## 🛠️ 開発者・メンテナー向け情報

- **ノートブックの一括再生成**:
  ```bash
  python3 scripts/gen_all.py
  ```
- **コードスタイル & プロットスタイル**:
  すべての可視化スクリプトおよびノートブックは `common.plot_utils.setup_style()` を適用し、論文水準の統一されたフォント・カラーパレットで描画されています。

---

## 📜 ライセンス

本リポジトリのコードは [MIT License](LICENSE) のもとで公開されています。
教材として自由にご活用ください。
