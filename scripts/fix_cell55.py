import nbformat

nb = nbformat.read('6/6_Exercises.ipynb', as_version=4)

nb.cells[54].source = r"""---
## Exercise 6.27: ラプラス近似 GPC における対数周辺尤度とその超パラメータ勾配

### 問題
ガウス過程分類におけるラプラス近似の対数周辺尤度（式 6.90）
$$
\log p(\mathbf{t}_N | \boldsymbol{\theta}) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} \log |\mathbf{I}_N + \mathbf{C}_N \mathbf{W}|
$$
を導出せよ。
さらに、カーネルの超パラメータ $\theta_j$ に関する対数尤度の勾配各項（式 6.91, 6.92, 6.94）
$$
\frac{d}{d\theta_j} \log p(\mathbf{t}_N | \boldsymbol{\theta}) = \frac{1}{2} \mathbf{a}_N^{*T} \mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \mathbf{C}_N^{-1} \mathbf{a}_N^* - \frac{1}{2} \text{Tr}\left( \mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \right) - \frac{1}{2} \frac{\partial}{\partial \theta_j} \log |\mathbf{I}_N + \mathbf{C}_N \mathbf{W}|
$$
を導出せよ。

### 数理的証明・導出ステップ（穴埋め）
1. 目的関数 $p(\mathbf{t}_N | \boldsymbol{\theta}) = \int p(\mathbf{t}_N | \mathbf{a}_N) p(\mathbf{a}_N | \boldsymbol{\theta}) d\mathbf{a}_N = \int \exp(\Psi(\mathbf{a}_N)) d\mathbf{a}_N$ に対し、モード $\mathbf{a}_N^*$ 周りで2次展開を行うと：
   $$
   \Psi(\mathbf{a}_N) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} (\mathbf{a}_N - \mathbf{a}_N^*)^T \mathbf{A} (\mathbf{a}_N - \mathbf{a}_N^*) \quad (\mathbf{A} = \mathbf{C}_N^{-1} + \mathbf{W})
   $$
2. ガウス積分を実行すると：
   $$
   \int \exp\left(-\frac{1}{2} (\mathbf{a} - \mathbf{a}^*)^T \mathbf{A} (\mathbf{a} - \mathbf{a}^*)\right) d\mathbf{a} = (2\pi)^{N/2} |\mathbf{A}|^{-1/2}
   $$
   したがって対数周辺尤度は：
   $$
   \log p(\mathbf{t}_N | \boldsymbol{\theta}) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} \log |\mathbf{A}| + \frac{N}{2} \log(2\pi)
   $$
   ここで事前分布の正規化項を含めて整理すると：
   $$
   |\mathbf{A}| |\mathbf{C}_N| = |(\mathbf{C}_N^{-1} + \mathbf{W}) \mathbf{C}_N| = |[ \text{①} ]|
   $$
   これにより式 (6.90) の $\log p(\mathbf{t}_N | \boldsymbol{\theta}) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} \log |\mathbf{I}_N + \mathbf{C}_N \mathbf{W}|$ が得られる。
3. **超パラメータ勾配**:
   全微分 $\frac{d}{d\theta_j} \log p$ をとるとき、モード点では $\nabla_{\mathbf{a}_N} \Psi(\mathbf{a}_N^*) = \mathbf{0}$ であるため、包絡線定理（Envelope Theorem）により $\Psi$ に対する陰関数微分項 $\frac{\partial \mathbf{a}_N^*}{\partial \theta_j}$ の寄与は厳密にゼロとなる。
   - $\Psi(\mathbf{a}_N^*)$ の陽な微分項：
     $$
     \frac{\partial \Psi}{\partial \theta_j} = \frac{1}{2} \mathbf{a}_N^{*T} \mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \mathbf{C}_N^{-1} \mathbf{a}_N^* - \frac{1}{2} \text{Tr}\left(\mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j}\right)
     $$
   - 行列式項 $-\frac{1}{2} \log |\mathbf{I} + \mathbf{C}_N \mathbf{W}|$ の微分：
     陽な微分項 $-\frac{1}{2} \text{Tr}\left( (\mathbf{I} + \mathbf{C}_N \mathbf{W})^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \mathbf{W} \right)$ に加え、$\mathbf{W}$ は $\mathbf{a}^*$ に依存するため、陰関数微分 $\frac{d\mathbf{a}^*}{d\theta_j} = (\mathbf{I} + \mathbf{C}_N \mathbf{W})^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \mathbf{C}_N^{-1} \mathbf{a}^*$ を通じた第3の寄与項（式 6.94）が含まれる。

### 穴埋めの解答
- ①: $\mathbf{I}_N + \mathbf{C}_N \mathbf{W}$
"""

nb.cells[55].source = r"""# Exercise 6.27 数値検証: ラプラス近似対数周辺尤度と超パラメータ解析的全勾配（式 6.91-6.94）の高精度一致
theta_val = 1.2

def compute_log_marginal_lik(th):
    C = th * rbf_mat(X, X) + 1e-3 * np.eye(N)
    a_mode = np.zeros(N)
    for _ in range(50):
        sg = sigmoid(a_mode)
        W_mat = np.diag(sg * (1.0 - sg))
        inv_m = np.linalg.inv(np.eye(N) + W_mat @ C)
        a_new = C @ inv_m @ (W_mat @ a_mode + t - sg)
        if np.max(np.abs(a_new - a_mode)) < 1e-12:
            a_mode = a_new
            break
        a_mode = a_new
    
    sg = sigmoid(a_mode)
    W_mat = np.diag(sg * (1.0 - sg))
    
    psi_val = -0.5 * a_mode @ np.linalg.solve(C, a_mode) - 0.5 * np.linalg.slogdet(C)[1] + \
              np.sum(t * np.log(np.clip(sg, 1e-12, 1-1e-12)) + (1-t) * np.log(np.clip(1-sg, 1e-12, 1-1e-12)))
    det_term = -0.5 * np.linalg.slogdet(np.eye(N) + C @ W_mat)[1]
    return psi_val + det_term, a_mode, C, W_mat

log_lik, a_opt, C_opt, W_opt = compute_log_marginal_lik(theta_val)
dC_dth = rbf_mat(X, X)
C_inv_a = np.linalg.solve(C_opt, a_opt)

# 項1: Psi の明示的 C 微分
term1 = 0.5 * C_inv_a @ dC_dth @ C_inv_a - 0.5 * np.trace(np.linalg.solve(C_opt, dC_dth))

# 項2: 行列式項の明示的 C 微分
inv_ICW = np.linalg.inv(np.eye(N) + C_opt @ W_opt)
term2 = -0.5 * np.trace(inv_ICW @ dC_dth @ W_opt)

# 項3: 行列式項の W(a*) 経由の陰関数微分 (da*/dtheta) (式 6.94)
da_dth = inv_ICW @ dC_dth @ C_inv_a
sg = sigmoid(a_opt)
dW_da = sg * (1.0 - sg) * (1.0 - 2.0 * sg)
inv_ICW_C = inv_ICW @ C_opt
term3 = -0.5 * np.sum(np.diag(inv_ICW_C) * dW_da * da_dth)

analytic_grad = term1 + term2 + term3

# 中心有限差分法による数値勾配
eps = 1e-6
log_lik_p, _, _, _ = compute_log_marginal_lik(theta_val + eps)
log_lik_m, _, _, _ = compute_log_marginal_lik(theta_val - eps)
numeric_grad = (log_lik_p - log_lik_m) / (2 * eps)

diff = abs(analytic_grad - numeric_grad)
assert np.isclose(analytic_grad, numeric_grad, rtol=1e-4, atol=1e-4)
print(f"Exercise 6.27 verified: Analytic Grad={analytic_grad:.6f} == Numeric Grad={numeric_grad:.6f} (Diff={diff:.2e})")
"""

nbformat.write(nb, '6/6_Exercises.ipynb')
print("Successfully fixed cells 54 and 55 in 6/6_Exercises.ipynb!")
