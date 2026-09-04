# PRML (Pattern Recognition and Machine Learning) Python Implementation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRML Coverage](https://img.shields.io/badge/Bishop%20PRML-Chapters%200--14%20(100%25)-brightgreen.svg)](TASK.md)
[![Tests](https://img.shields.io/badge/tests-53%20passed-success.svg)](tests/)

Christopher M. Bishop の世界的な名著 **『パターン認識と機械学習 (Pattern Recognition and Machine Learning, PRML)』** の理論、数式展開、およびアルゴリズムを Python / NumPy / SciPy によって忠実にスクラッチ実装し、インタラクティブな Jupyter Notebook と 118枚以上の高解像度可視化プロットで完全再現した学習・研究用リポジトリです。

---

## 🌟 主な特徴

1. **第0章から第14章までの完全網羅 (100% 達成)**
   - 確率論の基礎から最先端のグラフィカルモデル、変分推論、MCMC/HMCサンプリング、HMM、カルマンフィルタ、アンサンブル学習まで全65冊のノートブックで網羅。
2. **教科書図版の精緻な再現 (118+ Figures)**
   - PRML原著に登場するグラフ（多項式フィッティング、ガウス過程回帰、潜在空間、EMステップ、混合密度ネットワーク等）を各章の `result/` ディレクトリに忠実に再現・保存。
3. **穴埋め形式・証明ステップ付き章末演習問題 (Exercises)**
   - 全章について、初学者がモチベーションを維持しながら自力で数式展開を追体験できる論理ステップ提示＋穴埋め形式の章末演習ノートブック（`*_Exercises.ipynb`）を完備。
4. **体系的な Python パッケージ (`prml` / `common`)**
   - 全てのアルゴリズムを `prml.linear`, `prml.kernel`, `prml.clustering` などのサブパッケージに整理し、scikit-learn 風の洗練された API を提供（`pip install -e .` で利用可能）。
5. **包括的かつ高速な統合テストスイート (`tests/`)**
   - 53件の単体・統合テストにより、数理的整合性・境界条件・収束性を厳密に検証（約0.4秒で全件通過）。

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
├── tests/           # 統合・単体テストスイート (53 tests)
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
# 全15章・全モジュールの包括的ユニットテスト (53 tests, 約0.4秒で全件通過)
python3 -m unittest discover tests
```

---

## 📚 章別カリキュラムと主要トピック

| 章 | タイトル | 主なトピック・実装アルゴリズム | 再現図版 |
|---|---|---|:---:|
| **0** | **確率論の基礎** | 加法定理、乗法定理、ベイズの定理、変数変換定理、逆関数法サンプリング | 3 枚 |
| **1** | **序論** | 多項式曲線あてはめ、確率論基礎、モデル選択、次元の呪い、決定理論、損失関数、情報理論 | 20 枚 |
| **2** | **確率分布** | 二値・多項分布、Dirichlet、1次元・多次元ガウス分布、von Mises 分布、ノンパラメトリック密度推定 | 13 枚 |
| **3** | **線形回帰モデル** | 最小二乗法、正則化最小二乗、ベイズ線形回帰、エビデンス近似、バイアス-バリアンス分解 | 15 枚 |
| **4** | **線形分類モデル** | パーセプトロン、Fisher LDA、ロジスティック回帰、多クラスロジスティック、IRLS、ラプラス近似 | 12 枚 |
| **5** | **ニューラルネットワーク** | 多層パーセプトロン、誤差逆伝播法、数値勾配チェック、正則化、混合密度ネットワーク (MDN) | 7 枚 |
| **6** | **カーネル法** | 双対表現、RBF/ARDカーネル、Nadaraya-Watson 回帰、ガウス過程回帰 (GPR)、ガウス過程分類 (GPC) | 8 枚 |
| **7** | **スパースカーネルマシン** | サポートベクトルマシン (SVC)、SMO、関連ベクトルマシン (RVM 回帰・分類) | 6 枚 |
| **8** | **グラフィカルモデル** | ベイジアンネットワーク、d-分離、マルコフ確率場、Isingモデル、Factor Graph、Sum-Product法 | 3 枚 |
| **9** | **混合モデルとEM** | K-Means、混合ガウスモデル (GMM)、ベルヌーイ混合モデル、潜在変数とEMアルゴリズム | 6 枚 |
| **10** | **近似推論法** | 変分推論法 (Variational Inference)、変分混合ガウス (VB-GMM)、局所変分法 | 5 枚 |
| **11** | **サンプリング法** | 棄却サンプリング、重要度サンプリング、M-Hアルゴリズム、Gibbsサンプリング、ハミルトニアンモンテカルロ (HMC) | 6 枚 |
| **12** | **連続潜在変数** | 主成分分析 (PCA)、確率的主成分分析 (PPCA)、EMアルゴリズムによるPPCA、Kernel PCA | 5 枚 |
| **13** | **系列データ** | マルコフモデル、隠れマルコフモデル (HMM, Forward-Backward, Viterbi)、カルマンフィルタ (LDS) | 4 枚 |
| **14** | **モデルの結合** | バギング、ブースティング (AdaBoost)、決定株 (Decision Stump)、局所エキスパート混合 (MoE) | 5 枚 |

**合計: 118 枚の教科書再現図版を収録！**

---

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
