# scripts/enhance_ch10_exercises.py
"""
Master script to build, execute, and verify Chapter 10 exercises (10.1 to 10.39).
"""

import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

# Ensure scripts directory is in sys.path
sys.path.append(os.path.dirname(__file__))

from build_ch10_part1 import get_ex_10_1_to_10_9
from build_ch10_part2 import get_ex_10_10_to_10_19
from build_ch10_part3 import get_ex_10_20_to_10_28
from build_ch10_part4 import get_ex_10_29_to_10_39

def build_ch10_notebook():
    nb = nbf.v4.new_notebook()

    # Title & Table of Contents cell
    title_md = r"""# 第10章 近似推論 (Approximate Inference) — 演習問題完全解答・数値検証

本ノートブックは、C.M. Bishop 著『Pattern Recognition and Machine Learning (PRML)』**第10章「近似推論 (Approximate Inference)」**に収録されている **全39問（Exercises 10.1 〜 10.39）** の完全な論理証明、数理的思考プロセス、穴埋め形式のステップ導出、および自己完結した Python 数値検証コードを網羅した演習ノートブックです。

---

## 目次 (Table of Contents)

1. **変分推論の基礎、二変量ガウス因数分解、αダイバージェンス (Exercises 10.1 〜 10.9)**
   - [Exercise 10.1: 観測データの対数周辺尤度分解 (式 10.2 - 10.4)](#Exercise-10.1)
   - [Exercise 10.2: 二変量ガウス分布の因数分解変分近似 (式 10.13 - 10.15)](#Exercise-10.2)
   - [Exercise 10.3: KL(p || q) のラグランジュ乗数法による最小化 (式 10.16 - 10.17)](#Exercise-10.3)
   - [Exercise 10.4: ガウス分布による KL(p || q) 最小化 (モーメント整合)](#Exercise-10.4)
   - [Exercise 10.5: デルタ点推定近似変分法と EM アルゴリズムの等価性](#Exercise-10.5)
   - [Exercise 10.6: $\alpha$ ダイバージェンスの極限と KL ダイバージェンス (式 10.19)](#Exercise-10.6)
   - [Exercise 10.7: 1変量ガウス分布の因数分解変分更新式 (式 10.26 - 10.30)](#Exercise-10.7)
   - [Exercise 10.8: データ数大極限 $N \to \infty$ における変分精度事後分布の漸近挙動](#Exercise-10.8)
   - [Exercise 10.9: 期待精度の逆数に関する不動点方程式 (式 10.33)](#Exercise-10.9)

2. **変分モデル比較、変分混合ガウスモデル (V-GMM) (Exercises 10.10 〜 10.19)**
   - [Exercise 10.10: モデル事後分布の変分推論分解式 (10.34) の導出](#Exercise-10.10)
   - [Exercise 10.11: 変分下界最大化による最適モデル事後分布 $q^*(m) \propto p(m)\exp(\mathcal{L}_m)$ (10.36) の導出](#Exercise-10.11)
   - [Exercise 10.12: ベイズ混合ガウスにおける最適潜在変数事後分布 $q^*(\mathbf{Z})$ (10.48) の導出](#Exercise-10.12)
   - [Exercise 10.13: ガウス-ウィシャート変分事後分布 $q^*(\boldsymbol{\mu}_k, \mathbf{\Lambda}_k)$ (10.59) およびパラメータ更新式 (10.60-10.63) の導出](#Exercise-10.13)
   - [Exercise 10.14: ガウス-ウィシャート分布下の二次形式期待値公式 (10.64) の証明](#Exercise-10.14)
   - [Exercise 10.15: ディリクレ分布における混合比の期待値公式 (10.69) の証明](#Exercise-10.15)
   - [Exercise 10.16: ベイズ GMM 変分下界のデータ期待尤度項 (10.71) および潜在変数項 (10.72) の検証](#Exercise-10.16)
   - [Exercise 10.17: ベイズ GMM 変分下界の事前分布項およびエントロピー項 (10.73-10.77) の検証](#Exercise-10.17)
   - [Exercise 10.18: 変分下界 $\mathcal{L}$ の直接微分による GMM 再推定方程式の導出](#Exercise-10.18)
   - [Exercise 10.19: ベイズ混合ガウスの予測分布 $p(\widehat{\mathbf{x}}|\mathbf{X})$ が Student の t 分布混合となることの証明 (10.81)](#Exercise-10.19)

3. **大標本極限、対称性、特異点、ベイズ線形回帰変分推論、共役指数型分布族 (Exercises 10.20 〜 10.28)**
   - [Exercise 10.20: 大標本極限 $N \to \infty$ における変分ベイズ GMM の最尤推定 EM 解への漸近一致証明](#Exercise-10.20)
   - [Exercise 10.21: 混合モデルにおける置換対称性と $K!$ 個の同値モードの証明](#Exercise-10.21)
   - [Exercise 10.22: 単一モード変分近似と $K!$ 全対称化近似における事後予測分布の等価性](#Exercise-10.22)
   - [Exercise 10.23: 混合比 $\pi_k$ の点推定（事前分布なし）における変分下界最大化と最尤解一致](#Exercise-10.23)
   - [Exercise 10.24: 最大事後確率 (MAP) 推定における特異点再発と変分ベイズにおける特異点解消の比較論述](#Exercise-10.24)
   - [Exercise 10.25: 因数分解変分近似によるパラメータ共分散欠落と事後不確実性の過小評価](#Exercise-10.25)
   - [Exercise 10.26: 未知ノイズ精度 $\beta$ を含むベイズ線形回帰の変分推論更新式と下界の導出](#Exercise-10.26)
   - [Exercise 10.27: ベイズ線形回帰変分下界の各項 (10.107-10.112) の導出](#Exercise-10.27)
   - [Exercise 10.28: 共役指数型分布族の変分メッセージ伝播 (VMP) による GMM 更新式の統一的導出](#Exercise-10.28)

4. **局所変分法、Jaakkola-Jordan 境界、変分ロジスティック回帰、ADF & 期待伝播法 (EP) (Exercises 10.29 〜 10.39)**
   - [Exercise 10.29: 対数関数の凹性とルジャンドル変換による双対関数・元関数復元の証明](#Exercise-10.29)
   - [Exercise 10.30: 対数ロジスティック関数の凹性とテイラー展開による変分上界 (10.137) の導出](#Exercise-10.30)
   - [Exercise 10.31: Jaakkola-Jordan 下界関数 $\lambda(\xi)$ および二次形式下界 (10.144) の厳密な導出](#Exercise-10.31)
   - [Exercise 10.32: ベイズロジスティック回帰の逐次オンライン学習におけるガウス事後分布の閉包性](#Exercise-10.32)
   - [Exercise 10.33: 変分期待値関数 $\mathcal{Q}(\boldsymbol{\xi}, \boldsymbol{\xi}^{\mathrm{old}})$ の停留条件による $\xi_n^2$ 更新式 (10.163) の導出](#Exercise-10.33)
   - [Exercise 10.34: 変分下界 $\mathcal{L}(\boldsymbol{\xi})$ の直接最大化による再推定式の一致](#Exercise-10.34)
   - [Exercise 10.35: 変分ロジスティック回帰周辺下界 $\mathcal{L}(\boldsymbol{\xi})$ (10.164) のガウス積分による厳密導出](#Exercise-10.35)
   - [Exercise 10.36: Assumed Density Filtering (ADF) におけるモデルエビデンス逐次更新則 (10.242) の証明](#Exercise-10.36)
   - [Exercise 10.37: EP における共役事前分布因子の不動点不変性の証明](#Exercise-10.37)
   - [Exercise 10.38: クラッター問題における EP キャビティ分布 (10.214-10.215) および規格化定数 $Z_n$ (10.216) の導出](#Exercise-10.38)
   - [Exercise 10.39: クラッター問題における EP 更新平均 $\mathbf{m}^{\mathrm{new}}$ および分散 $v^{\mathrm{new}}$ (10.217-10.218) の導出](#Exercise-10.39)

---
"""

    setup_code = r"""# 共通ライブラリのインポートとパス設定
import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import scipy
from scipy import integrate
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, wishart, gamma as gamma_dist
from scipy.special import psi, gamma, gammaln
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Environment successfully initialized for Chapter 10 exercises.")"""

    nb.cells.append(nbf.v4.new_markdown_cell(title_md))
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Append all parts
    c1 = get_ex_10_1_to_10_9()
    c2 = get_ex_10_10_to_10_19()
    c3 = get_ex_10_20_to_10_28()
    c4 = get_ex_10_29_to_10_39()

    nb.cells.extend(c1)
    nb.cells.extend(c2)
    nb.cells.extend(c3)
    nb.cells.extend(c4)

    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../10/10_Exercises.ipynb"))
    print(f"Total cells to execute: {len(nb.cells)}")

    ep = ExecutePreprocessor(timeout=180, kernel_name='python3')
    print("Executing Chapter 10 notebook cells via ExecutePreprocessor...")
    ep.preprocess(nb, {'metadata': {'path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../10/'))}})
    print("Execution completed successfully!")

    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Saved executed notebook to: {target_path}")

if __name__ == "__main__":
    build_ch10_notebook()
