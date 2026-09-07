# scripts/enhance_ch9_exercises.py
"""
Master script to build, execute, and verify Chapter 9 exercises (9.1 to 9.27).
"""

import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from build_ch9_part1 import get_ex_9_1_to_9_9
from build_ch9_part2 import get_ex_9_10_to_9_18
from build_ch9_part3 import get_ex_9_19_to_9_27

def build_ch9_notebook():
    nb = nbf.v4.new_notebook()

    # Title & Overview cell
    title_md = r"""# 第9章 混合モデルとEMアルゴリズム (Mixture Models and EM) — 演習問題完全解答・数値検証

本ノートブックは、C.M. Bishop 著『Pattern Recognition and Machine Learning (PRML)』**第9章「混合モデルとEMアルゴリズム (Mixture Models and EM)」**に収録されている **全27問（Exercises 9.1 〜 9.27）** の完全な論理証明、数式ステップ穴埋め問題、および自己完結した Python 数値検証コードを網羅した演習ノートブックです。

---

## 目次 (Table of Contents)

1. **K-means、基本GMMおよびパラメータ事前分布 (Exercises 9.1 〜 9.9)**
   - [Exercise 9.1: K-means アルゴリズムの有限回反復収束証明](#Exercise-9.1)
   - [Exercise 9.2: Robbins-Monro 逐次推定法によるオンライン K-means 更新則の導出](#Exercise-9.2)
   - [Exercise 9.3: ガウス混合モデルにおける周辺分布と事後負担率（ベイズの定理）の導出](#Exercise-9.3)
   - [Exercise 9.4: パラメータ事前分布を持つモデルにおける MAP 推定 EM アルゴリズム](#Exercise-9.4)
   - [Exercise 9.5: d分離による潜在変数事後分布のデータ点間因数分解証明](#Exercise-9.5)
   - [Exercise 9.6: 共通共分散行列 $\mathbf{\Sigma}_k = \mathbf{\Sigma}$ を持つ GMM の EM アルゴリズム導出](#Exercise-9.6)
   - [Exercise 9.7: 完全データ対数尤度最大化による各成分パラメータの独立最尤推定](#Exercise-9.7)
   - [Exercise 9.8: 期待完全データ対数尤度の中心ベクトル $\boldsymbol{\mu}_k$ に関する閉形式最大化](#Exercise-9.8)
   - [Exercise 9.9: 期待完全データ対数尤度の共分散 $\mathbf{\Sigma}_k$ および混合係数 $\pi_k$ に関する閉形式最大化](#Exercise-9.9)

2. **一般混合モデル、ベルヌーイ混合モデル (BMM) (Exercises 9.10 〜 9.18)**
   - [Exercise 9.10: 分割ベクトルにおける条件付き分布の混合分布表現](#Exercise-9.10)
   - [Exercise 9.11: 分散ゼロ極限 $\epsilon \to 0$ における EM アルゴリズムと K-means の等価性](#Exercise-9.11)
   - [Exercise 9.12: 一般混合分布全体の平均および共分散行列の合成則 (式 9.49, 9.50)](#Exercise-9.12)
   - [Exercise 9.13: ベルヌーイ混合モデルにおける同一初期値での1反復退化現象の証明](#Exercise-9.13)
   - [Exercise 9.14: ベルヌーイ混合モデルの同時分布と潜在変数周辺化による観測分布の導出](#Exercise-9.14)
   - [Exercise 9.15: ベルヌーイ混合モデルにおける中心パラメータ $\boldsymbol{\mu}_k$ の M ステップ再推定式 (9.59) の導出](#Exercise-9.15)
   - [Exercise 9.16: ベルヌーイ混合モデルにおける混合係数 $\pi_k$ の M ステップ更新式 (9.60) の導出](#Exercise-9.16)
   - [Exercise 9.17: ベルヌーイ混合モデルにおける対数尤度の上界性と特異点（発散）の非存在証明](#Exercise-9.17)
   - [Exercise 9.18: Beta-Dirichlet 事前分布を持つベルヌーイ混合モデルの MAP-EM アルゴリズム導出](#Exercise-9.18)

3. **多項分布混合、回帰EM、変分下界の幾何、インクリメンタルEM (Exercises 9.19 〜 9.27)**
   - [Exercise 9.19: 多項分布混合モデル (Mixture of Multinomials) の EM アルゴリズム導出](#Exercise-9.19)
   - [Exercise 9.20: ベイズ線形回帰における超パラメータ $\alpha$ の EM アルゴリズム再推定式 (9.63) の導出](#Exercise-9.20)
   - [Exercise 9.21: ベイズ線形回帰におけるノイズ精度 $\beta$ の EM アルゴリズム再推定式の導出](#Exercise-9.21)
   - [Exercise 9.22: 関連ベクトルマシン (RVM) における超パラメータ再推定式 (9.67, 9.68) の EM 導出](#Exercise-9.22)
   - [Exercise 9.23: RVM における周辺尤度直接最大化と EM アルゴリズムの形式的等価性証明](#Exercise-9.23)
   - [Exercise 9.24: 対数尤度の変分分解 $\ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \| p)$ の証明](#Exercise-9.24)
   - [Exercise 9.25: 接点における変分下界と対数尤度の勾配一致定理の証明](#Exercise-9.25)
   - [Exercise 9.26: オンライン（インクリメンタル）EM における中心ベクトル逐次更新式 (9.78, 9.79) の導出](#Exercise-9.26)
   - [Exercise 9.27: インクリメンタル EM における共分散行列 $\mathbf{\Sigma}_k$ および混合係数 $\pi_k$ の逐次更新式の導出](#Exercise-9.27)

---
"""

    setup_code = r"""# 共通ライブラリのインポートとパス設定
import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Environment successfully initialized for Chapter 9 exercises.")"""

    nb.cells.append(nbf.v4.new_markdown_cell(title_md))
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Append all parts
    nb.cells.extend(get_ex_9_1_to_9_9())
    nb.cells.extend(get_ex_9_10_to_9_18())
    nb.cells.extend(get_ex_9_19_to_9_27())

    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../9/9_Exercises.ipynb"))
    print(f"Total cells to execute: {len(nb.cells)}")

    ep = ExecutePreprocessor(timeout=120, kernel_name='python3')
    print("Executing Chapter 9 notebook cells via ExecutePreprocessor...")
    ep.preprocess(nb, {'metadata': {'path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../9/'))}})
    print("Execution completed successfully!")

    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Saved executed notebook to: {target_path}")

if __name__ == "__main__":
    build_ch9_notebook()
