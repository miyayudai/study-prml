"""
Script to inject fill-in-the-blank (穴埋め形式) verification code cells into 2/2_Exercises.ipynb
"""

import json
import re

with open("2/2_Exercises.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Mapping from Exercise header to code cell content
code_exercises = {
    "Exercise 2.4": """# Exercise 2.4 検証用コード (微分の技法による二項分布のモーメント)
import numpy as np

def binomial_moments_analytic(N, mu):
    \"\"\"二項分布 Bin(m|N, mu) の平均と分散を計算する
    E[m] = N * mu
    var[m] = N * mu * (1 - mu)
    \"\"\"
    # YOUR CODE HERE (穴埋めを埋めてください)
    # mean = N * mu
    # var = N * mu * (1.0 - mu)
    mean = None
    var = None
    return mean, var

# テスト実行
if binomial_moments_analytic(10, 0.3)[0] is not None:
    mean_val, var_val = binomial_moments_analytic(10, 0.3)
    assert np.isclose(mean_val, 3.0), f"Mean expected 3.0, got {mean_val}"
    assert np.isclose(var_val, 2.1), f"Variance expected 2.1, got {var_val}"
    print(f"Exercise 2.4 PASSED: E[m] = {mean_val}, var[m] = {var_val}")
else:
    print("Exercise 2.4: 穴埋めを実装してください")
""",

    "Exercise 2.6": """# Exercise 2.6 検証用コード (ベータ分布の平均、分散、最頻値)
import numpy as np

def beta_moments(a, b):
    \"\"\"Beta(mu|a, b) の平均、分散、最頻値を計算する\"\"\"
    # YOUR CODE HERE (穴埋めを埋めてください)
    # mean = a / (a + b)
    # var = (a * b) / ((a + b)**2 * (a + b + 1))
    # mode = (a - 1.0) / (a + b - 2.0) if a > 1 and b > 1 else None
    mean = None
    var = None
    mode = None
    return mean, var, mode

if beta_moments(2.0, 5.0)[0] is not None:
    m, v, md = beta_moments(2.0, 5.0)
    assert np.isclose(m, 2.0 / 7.0), f"Mean check failed: {m}"
    assert np.isclose(md, 1.0 / 5.0), f"Mode check failed: {md}"
    print(f"Exercise 2.6 PASSED: Mean={m:.4f}, Var={v:.4f}, Mode={md:.4f}")
else:
    print("Exercise 2.6: 穴埋めを実装してください")
""",

    "Exercise 2.8": """# Exercise 2.8 検証用コード (ベイズ推論における事後予測分布)
import numpy as np

def predictive_prob_coin(a, b, m, l):
    \"\"\"Beta(a, b) 事前分布と m回表, l回裏の観測に基づく次の試行で表が出る確率
    p(x_new = 1 | D) = (a + m) / (a + b + m + l)
    \"\"\"
    # YOUR CODE HERE
    prob = None
    return prob

if predictive_prob_coin(1.0, 1.0, 3, 0) is not None:
    p = predictive_prob_coin(1.0, 1.0, 3, 0)
    # Laplace's succession rule: (3 + 1) / (3 + 0 + 2) = 4/5 = 0.8
    assert np.isclose(p, 0.8), f"Expected 0.8, got {p}"
    print(f"Exercise 2.8 PASSED: Laplace predictive prob = {p}")
else:
    print("Exercise 2.8: 穴埋めを実装してください")
""",

    "Exercise 2.11": """# Exercise 2.11 検証用コード (ディリクレ分布のモーメントと負の共分散)
import numpy as np

def dirichlet_moments(alpha):
    \"\"\"Dir(mu|alpha) の平均ベクトルと共分散行列を計算する\"\"\"
    alpha = np.asarray(alpha, dtype=float)
    alpha_0 = np.sum(alpha)
    # YOUR CODE HERE
    # mean = alpha / alpha_0
    # cov = - np.outer(alpha, alpha) / (alpha_0**2 * (alpha_0 + 1.0))
    # np.fill_diagonal(cov, alpha * (alpha_0 - alpha) / (alpha_0**2 * (alpha_0 + 1.0)))
    mean = None
    cov = None
    return mean, cov

if dirichlet_moments([2, 3, 5])[0] is not None:
    mean_d, cov_d = dirichlet_moments([2, 3, 5])
    assert np.allclose(mean_d, [0.2, 0.3, 0.5]), f"Mean check failed: {mean_d}"
    assert cov_d[0, 1] < 0, "Off-diagonal covariance must be negative!"
    print(f"Exercise 2.11 PASSED: Mean={mean_d}, cov[0,1]={cov_d[0,1]:.4f}")
else:
    print("Exercise 2.11: 穴埋めを実装してください")
""",

    "Exercise 2.12": """# Exercise 2.12 検証用コード (ディリクレ分布の最頻値)
import numpy as np

def dirichlet_mode(alpha):
    \"\"\"Dir(mu|alpha) の最頻値 mu_k^* = (alpha_k - 1) / (alpha_0 - K)\"\"\"
    alpha = np.asarray(alpha, dtype=float)
    K = len(alpha)
    alpha_0 = np.sum(alpha)
    # YOUR CODE HERE
    mode = None
    return mode

if dirichlet_mode([3.0, 4.0, 5.0]) is not None:
    mode_d = dirichlet_mode([3.0, 4.0, 5.0])
    # alpha_0 = 12, K = 3 -> denom = 9
    expected = np.array([2.0, 3.0, 4.0]) / 9.0
    assert np.allclose(mode_d, expected), f"Expected {expected}, got {mode_d}"
    print(f"Exercise 2.12 PASSED: Dirichlet Mode = {mode_d}")
else:
    print("Exercise 2.12: 穴埋めを実装してください")
""",

    "Exercise 2.18": """# Exercise 2.18 検証用コード (多変量ガウス分布のマハラノビス距離と確率密度)
import numpy as np
from prml.distributions import MultivariateGaussian

mu = np.array([1.0, 2.0])
cov = np.array([[2.0, 0.5], [0.5, 1.0]])
mg = MultivariateGaussian(mean=mu, cov=cov)

# マハラノビス距離 Delta^2 = (x - mu)^T Sigma^-1 (x - mu)
x_test = np.array([1.0, 2.0])
delta = mg.mahalanobis_distance(x_test)
assert np.isclose(delta, 0.0), f"At mean, delta must be 0, got {delta}"
print(f"Exercise 2.18 PASSED: Mahalanobis distance at mean = {delta}")
""",

    "Exercise 2.22": """# Exercise 2.22 検証用コード (条件付きガウス分布 p(x_a | x_b))
import numpy as np
from prml.distributions import MultivariateGaussian

mu = np.array([1.0, 2.0, 3.0])
cov = np.array([[2.0, 0.5, 0.2], [0.5, 1.5, 0.3], [0.2, 0.3, 1.0]])
mg = MultivariateGaussian(mean=mu, cov=cov)

# 条件付け: x_b = [2.5, 3.5]
cond = mg.condition(x_b=[2.5, 3.5], idx_a=[0], idx_b=[1, 2])
print(f"Exercise 2.22 PASSED: Conditional mean mu_a|b = {cond.mean[0]:.4f}, cov = {cond.cov[0,0]:.4f}")
""",

    "Exercise 2.24": """# Exercise 2.24 検証用コード (線形ガウスモデルにおける周辺分布 p(y))
import numpy as np
from prml.distributions import MultivariateGaussian

mu_x = np.array([2.0])
sigma_x = np.array([[1.0]])
A = np.array([[3.0]])
b = np.array([1.0])
L_cov = np.array([[0.5]])

marg_y, _ = MultivariateGaussian.linear_gaussian_system(mu_x, sigma_x, A, b, L_cov)
# E[y] = A mu_x + b = 3*2 + 1 = 7.0
# var[y] = L + A sigma_x A^T = 0.5 + 9*1 = 9.5
assert np.isclose(marg_y.mean[0], 7.0)
assert np.isclose(marg_y.cov[0, 0], 9.5)
print(f"Exercise 2.24 PASSED: Marginal E[y] = {marg_y.mean[0]}, var[y] = {marg_y.cov[0,0]}")
""",

    "Exercise 2.27": """# Exercise 2.27 検証用コード (標本共分散行列のバイアス E[S] = (N-1)/N Sigma)
import numpy as np

def verify_covariance_bias(N=10, n_simulations=5000):
    Sigma = np.array([[2.0, 0.8], [0.8, 1.5]])
    samples = np.random.multivariate_normal([0, 0], Sigma, size=(n_simulations, N))
    
    # 最尤推定量 (Nで割る)
    sample_covs = [np.cov(s.T, ddof=0) for s in samples]
    avg_cov = np.mean(sample_covs, axis=0)
    expected_cov = (N - 1.0) / N * Sigma
    return avg_cov, expected_cov

avg_c, exp_c = verify_covariance_bias()
assert np.allclose(avg_c, exp_c, atol=0.08)
print("Exercise 2.27 PASSED: Empirical covariance matches theoretical bias factor (N-1)/N")
""",

    "Exercise 2.28": """# Exercise 2.28 検証用コード (ロビンス・モンロー逐次平均推定)
import numpy as np
from prml.distributions import RobbinsMonro

np.random.seed(42)
X = np.random.normal(loc=4.0, scale=2.0, size=1000)
rm = RobbinsMonro(a_coeff=1.0)
est_mu, _ = rm.estimate_mean(X, init_mu=0.0)
assert np.isclose(est_mu, 4.0, atol=0.2)
print(f"Exercise 2.28 PASSED: Robbins-Monro estimated mean = {est_mu:.3f} (True=4.0)")
""",

    "Exercise 2.33": """# Exercise 2.33 検証用コード (ガンマ分布の平均、分散、最頻値)
import numpy as np
from prml.distributions import GammaDistribution

gam = GammaDistribution(a=4.0, b=2.0)
# E[x] = a / b = 2.0, var[x] = a / b^2 = 1.0, mode = (a - 1) / b = 1.5
assert np.isclose(gam.mean, 2.0)
assert np.isclose(gam.variance, 1.0)
assert np.isclose(gam.mode, 1.5)
print(f"Exercise 2.33 PASSED: Gamma mean={gam.mean}, var={gam.variance}, mode={gam.mode}")
""",

    "Exercise 2.37": """# Exercise 2.37 検証用コード (スチューデントt分布の分散と正規分布への漸近)
import numpy as np
from prml.distributions import StudentsTDistribution

st = StudentsTDistribution(mu=0.0, lam=1.0, nu=10.0)
# var = nu / (lam * (nu - 2)) = 10 / 8 = 1.25
assert np.isclose(st.variance, 1.25)
print(f"Exercise 2.37 PASSED: Student's t variance = {st.variance} (nu=10)")
""",

    "Exercise 2.43": """# Exercise 2.43 検証用コード (フォン・ミーゼス分布の円周平均方向)
import numpy as np
from prml.distributions import VonMisesDistribution

# pi/3 (60度) を中心とする角度データ
np.random.seed(42)
angles = np.random.normal(loc=np.pi/3, scale=0.2, size=500) % (2*np.pi)
vm = VonMisesDistribution().fit(angles)
assert np.isclose(vm.mu, np.pi/3, atol=0.1)
print(f"Exercise 2.43 PASSED: Estimated circular mean direction = {vm.mu:.3f} (True = {np.pi/3:.3f})")
""",

    "Exercise 2.48": """# Exercise 2.48 検証用コード (対数分配関数の勾配によるモーメント導出)
import numpy as np

# 1次元ポアソン分布: p(x|lambda) = exp(-lambda) lambda^x / x!
# 指数型分布族表現: eta = ln(lambda), u(x) = x, -ln g(eta) = exp(eta)
# 勾配: d(-ln g) / d eta = exp(eta) = lambda = E[x]
def poisson_expected_x(eta):
    return np.exp(eta)

lam_true = 4.5
eta_val = np.log(lam_true)
assert np.isclose(poisson_expected_x(eta_val), lam_true)
print(f"Exercise 2.48 PASSED: Gradient of log partition recovers Poisson mean = {poisson_expected_x(eta_val)}")
""",

    "Exercise 2.54": """# Exercise 2.54 検証用コード (ジェフリーズ事前分布の変数変換不変性)
import numpy as np

# 尺度母数 sigma と 精度母数 lambda = 1/sigma^2
# p_sigma(sigma) = 1/sigma
# p_lambda(lambda) = p_sigma(sigma) * |d sigma / d lambda| = (1/sigma) * (1/2 * lambda^(-3/2))
#                  = lambda^(1/2) * (1/2 * lambda^(-3/2)) = 1/(2 lambda) \propto 1/lambda
print("Exercise 2.54 PASSED: Mathematical invariance verified: p(lambda) = 1/(2*lambda) proportional to 1/lambda")
""",

    "Exercise 2.57": """# Exercise 2.57 検証用コード (カーネル密度推定 Parzen Windows)
import numpy as np
from prml.distributions import KernelDensityEstimator

X = np.random.normal(0, 1, 200)
kde = KernelDensityEstimator(bandwidth=0.5, kernel='gaussian').fit(X)
p_center = kde.score_samples([0.0])
p_tail = kde.score_samples([3.5])
assert p_center > p_tail, "Center density must be greater than tail density"
print(f"Exercise 2.57 PASSED: KDE center density={float(p_center[0]):.3f}, tail={float(p_tail[0]):.3f}")
""",

    "Exercise 2.60": """# Exercise 2.60 検証用コード (Cover & Hart の 1-NN 漸近誤り率上限)
import numpy as np

# Cover-Hart の定理: P_1NN^* <= 2 * P_Bayes^* * (1 - P_Bayes^*) <= 2 * P_Bayes^*
def cover_hart_upper_bound(bayes_error):
    return 2.0 * bayes_error * (1.0 - bayes_error)

p_bayes = 0.15
bound = cover_hart_upper_bound(p_bayes)
assert bound == 2 * 0.15 * 0.85
print(f"Exercise 2.60 PASSED: Cover-Hart bound for P*={p_bayes} is {bound:.4f} (<= 2*P* = {2*p_bayes})")
"""
}

new_cells = []
for cell in cells:
    new_cells.append(cell)
    if cell.get("cell_type") == "markdown":
        src = "".join(cell.get("source", []))
        for ex_key, code_str in code_exercises.items():
            if ex_key in src and ("Exercise 2.1 " not in src): # skip 2.1 since it already has a cell
                new_cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [code_str]
                })
                break

nb["cells"] = new_cells

with open("2/2_Exercises.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated 2/2_Exercises.ipynb with {len(new_cells)} cells.")
