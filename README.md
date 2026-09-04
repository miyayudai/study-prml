# Pattern Recognition and Machine Learning (PRML) - Complete Python Implementation & Exercise Solutions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![PRML Coverage](https://img.shields.io/badge/PRML%20Chapters-0%20to%2014%20(100%25)-orange.svg)]()
[![Notebooks](https://img.shields.io/badge/Jupyter%20Notebooks-63%20Total-blueviolet.svg)]()
[![Reproduced Figures](https://img.shields.io/badge/PRML%20Figures-101%20Reproduced-success.svg)]()

Christopher M. Bishop 著の名著 **『Pattern Recognition and Machine Learning』(PRML)** の**全14章（第1章〜第14章）および準備章（第0章：確率・確率密度の基礎補講）**を完全に網羅した、Pythonによる数理アルゴリズムのスクラッチ実装・原著図版の完全再現・および**全章演習問題（Exercises）の詳細な数学的証明と数値検証コード**を収録した包括的リポジトリです。

---

## 🌟 プロジェクトの特色

1. **完全スクラッチ実装 (Zero-Black-Box)**
   - 機械学習の「ブラックボックス」を排し、NumPy / SciPy による線形代数・微積分・凸最適化の基礎方程式からすべてのモデル（ベイズ線形回帰、ガウス過程、SVM/SMO、RVM、EM、変分推論、HMC、PPCA、HMM、カルマンフィルタ、AdaBoost等）を忠実に実装しています。
2. **scikit-learn 準拠の統一 API (`fit` / `predict` / `transform`)**
   - 共通モジュール [`common/`](file:///home/student/Documents/GitHub/my_PRML/common) 配下に再利用可能なクラス群を設計し、一貫したインターフェースを提供。
3. **原著図版の忠実な再現（計 101 枚）**
   - 本文中に登場する象徴的なグラフ（ベイズ予測分布、ガウス過程サンプルパス、SMO決定境界、GMM特異点と収束過程、変分下界の単調増加と不要成分の自動消滅、HMC位相空間、固有数字と段階的再構成、カルマン不確実性拡散と収縮等）を高解像度で再現し、各章の `result/` に保存。
4. **全演習問題（Exercises）の完全網羅**
   - ラグランジュ未定乗数法、変分法、d-分離、情報理論的不等式、ベイズ更新公式の厳密な数理証明に加え、Python による直接数値計算・シミュレーション検証を全問完備。

---

## 📚 各章の構成と実装内容一覧 (Table of Contents)

| 章 | タイトル (Title) | ノートブック数 | 再現図版数 | 主な実装アルゴリズム & トピック | 演習問題 |
|:---:|:---|:---:|:---:|:---|:---:|
| **[Ch 0](file:///home/student/Documents/GitHub/my_PRML/0)** | **確率・確率密度の基礎補講**<br>*(Foundations of Probability)* | 2 | 3 | ヤコビアンと確率密度の変数変換定理、累積分布関数（CDF）逆変換サンプリング法 | 全問検証 |
| **[Ch 1](file:///home/student/Documents/GitHub/my_PRML/1)** | **序論**<br>*(Introduction)* | 5 | 3 | 多項式曲線フィッティング、過学習と過小学習、正則化項の効果、ベイズ決定理論、情報理論とエントロピー | 1.1 - 1.41 |
| **[Ch 2](file:///home/student/Documents/GitHub/my_PRML/2)** | **確率分布**<br>*(Probability Distributions)* | 6 | 13 | ガウス分布の幾何学・条件付き/周辺化、ガンマ・ベータ・ディリクレ共役事前分布、スチューデントのt分布、指数型分布族 | 2.1 - 2.61 |
| **[Ch 3](file:///home/student/Documents/GitHub/my_PRML/3)** | **線形回帰モデル**<br>*(Linear Models for Regression)* | 6 | 15 | 基底関数展開、バイアス-バリアンス分解、ベイズ線形回帰と逐次事後分布更新、等価カーネル、エビデンス近似 | 3.1 - 3.24 |
| **[Ch 4](file:///home/student/Documents/GitHub/my_PRML/4)** | **線形分類モデル**<br>*(Linear Models for Classification)* | 5 | 12 | フィッシャーの線形判別 (LDA)、パーセプトロン収束定理、ロジスティック回帰 (IRLS)、プロビット回帰、ラプラス近似 | 4.1 - 4.26 |
| **[Ch 5](file:///home/student/Documents/GitHub/my_PRML/5)** | **ニューラルネットワーク**<br>*(Neural Networks)* | 5 | 7 | 多層パーセプトロン、誤差逆伝播法（Backpropagation）、数値微分勾配検証、ヘシアン解析、混合密度ネットワーク (MDN) | 5.1 - 5.41 |
| **[Ch 6](file:///home/student/Documents/GitHub/my_PRML/6)** | **カーネル法**<br>*(Kernel Methods)* | 5 | 8 | 双対表現とカーネルトリック、カーネルリッジ回帰、ナダラヤ・ワトソン核回帰、ガウス過程回帰 (GPR) / 分類 (GPC) | 6.1 - 6.27 |
| **[Ch 7](file:///home/student/Documents/GitHub/my_PRML/7)** | **疎なカーネルマシン**<br>*(Sparse Kernel Machines)* | 3 | 6 | サポートベクトルマシン (SVM / SMO アルゴリズム)、KKT 相補性条件、関連ベクトルマシン (RVM 回帰 & 分類) | 7.1 - 7.19 |
| **[Ch 8](file:///home/student/Documents/GitHub/my_PRML/8)** | **グラフィカルモデル**<br>*(Graphical Models)* | 4 | 3 | 有向分離 (d-separation)、マルコフ確率場 (MRF) とクリークポテンシャル、因子グラフ、確率伝播 (Sum-Product アルゴリズム) | 8.1 - 8.29 |
| **[Ch 9](file:///home/student/Documents/GitHub/my_PRML/9)** | **混合モデルとEM**<br>*(Mixture Models and EM)* | 4 | 6 | K-means、ガウス混合モデル (GMM)、ベルヌーイ混合モデル、最尤特異点回避、EMアルゴリズムの幾何学 | 9.1 - 9.27 |
| **[Ch 10](file:///home/student/Documents/GitHub/my_PRML/10)** | **近似推論法**<br>*(Approximate Inference)* | 4 | 5 | 平均場変分推論、変分ベイズ GMM (VB-GMM)、不要クラスタの自律的消滅、変分下界 $\mathcal{L}(q)$ の単調収束、EP法 | 10.1 - 10.39 |
| **[Ch 11](file:///home/student/Documents/GitHub/my_PRML/11)** | **サンプリング法**<br>*(Sampling Methods)* | 4 | 6 | 採択サンプリング、重点サンプリング、SIR、メトロポリス・ヘイスティングス (M-H)、ギブスサンプリング、HMC (Leapfrog法) | 11.1 - 11.17 |
| **[Ch 12](file:///home/student/Documents/GitHub/my_PRML/12)** | **連続潜在変数**<br>*(Continuous Latent Variables)* | 4 | 5 | 主成分分析 (最大分散 & 最小再構成誤差)、手書き数字固有画像、白色化変換、確率的PCA (EM & 閉形式解)、カーネルPCA | 12.1 - 12.29 |
| **[Ch 13](file:///home/student/Documents/GitHub/my_PRML/13)** | **系列データ**<br>*(Sequential Data)* | 3 | 4 | ガウス放出 HMM、Forward-Backward アルゴリズム (スケーリング係数 $c_n$)、ビタビアルゴリズム、カルマンフィルタ & RTS スムーザ | 13.1 - 13.34 |
| **[Ch 14](file:///home/student/Documents/GitHub/my_PRML/14)** | **モデル結合**<br>*(Combining Models)* | 3 | 5 | コミッティ / バギング分散低減定理、AdaBoost、各種サロゲート損失関数比較、決定木不純度、線形回帰混合モデル (MoE) | 14.1 - 14.17 |
| **合計** | **全15ディレクトリ** | **63冊** | **101枚** | **数理的アルゴリズム・グラフィカルモデル・サンプリング・深層生成の完全実装** | **全390+問 完備** |

---

## 🛠️ 共通モジュール (`common/`)

本リポジトリでは重複のないクリーンなコードベースを維持するため、全章で利用される基盤ロジックを [`common/`](file:///home/student/Documents/GitHub/my_PRML/common) にモジュール化しています：

```python
from common import (
    # 線形回帰 & 分類
    LinearRegression, BayesianLinearRegression, EvidenceApproximation,
    LogisticRegression, MulticlassLogisticRegression,
    # ニューラルネット & カーネル法
    MLPRegressor, MixtureDensityNetwork,
    GaussianProcessRegressor, GaussianProcessClassifier,
    # 疎なカーネルマシン
    SupportVectorClassifier, RelevanceVectorRegressor,
    # グラフィカルモデル & クラスタリング
    SimpleFactorGraphChain, KMeans, GaussianMixtureModel,
    # 変分推論 & サンプリング
    VariationalGaussianMixture,
    rejection_sample, metropolis_hastings, hamiltonian_monte_carlo,
    # 潜在変数モデル & 系列データ & アンサンブル
    PCA, ProbabilisticPCA, KernelPCA,
    GaussianHMM, KalmanFilter,
    DecisionStump, AdaBoostClassifier, MixtureOfLinearRegressions,
    # 可視化ユーティリティ
    setup_style, save_plot
)
```

---

## 🚀 クイックスタート (Environment & Usage)

### 1. 環境構築

Python 3.10 以上がインストールされた環境で以下を実行します：

```bash
# リポジトリのクローン
git clone https://github.com/miyayudai/my_PRML.git
cd my_PRML

# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 統合テストスイートの実行

実装された主要アルゴリズムの動作検証を一括実行できます：

```bash
python3 -m unittest discover tests
```

### 3. Jupyter Notebook の起動

```bash
jupyter lab
# または
jupyter notebook
```

任意の章（例: `3/3.3_Bayesian_Linear_Regression.ipynb` や `13/13.1-13.2_Hidden_Markov_Models.ipynb`）を開いて実行してください。

---

## 📁 ディレクトリ構造

```text
my_PRML/
├── 0/                # 第0章 確率・確率密度の基礎補講 (ノートブック & result/)
├── 1/ 〜 14/         # 第1章〜第14章の各論ノートブック & 演習問題 (Exercises)
│   ├── *.ipynb       # 理論数式・スクラッチ実装・考察
│   └── result/       # 生成された PRML 再現高解像度グラフ (*.png)
├── common/           # 共通機械学習アルゴリズム & プロットモジュール
│   ├── __init__.py   # 統一エクスポート
│   ├── regression_utils.py
│   ├── classification_utils.py
│   ├── nn_utils.py
│   ├── kernel_utils.py
│   ├── svm_rvm_utils.py
│   ├── graphical_models_utils.py
│   ├── mixture_em_utils.py
│   ├── variational_utils.py
│   ├── sampling_utils.py
│   ├── pca_ppca_utils.py
│   ├── sequential_utils.py
│   ├── ensemble_utils.py
│   └── plot_utils.py
├── scripts/          # ノートブック自動生成スクリプト群
├── tests/            # 統合ユニットテストスイート
├── TASK.md           # 進捗管理・タスクリスト (100% Complete)
├── requirements.txt  # 依存パッケージ一覧
└── README.md         # 本ドキュメント
```

---

## 📖 参考文献

- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
  (原著 PDF: [Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf](file:///home/student/Documents/GitHub/my_PRML/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf))
