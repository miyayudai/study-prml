import numpy as np
import scipy.special as sp_special

def tanh(a):
    return np.tanh(a)

def dtanh(a):
    return 1.0 - np.tanh(a)**2

def softmax(a, axis=-1):
    a_max = np.max(a, axis=axis, keepdims=True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis=axis, keepdims=True)

class MLPRegressor:
    """
    2層フィードフォワードニューラルネットワーク (回帰用, PRML 5.1-5.3節)
    入力 -> 隠れ層 (tanh) -> 出力層 (線形)
    """
    def __init__(self, n_in, n_hidden, n_out, weight_decay=0.0, lr=0.01, random_state=42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.weight_decay = weight_decay
        self.lr = lr
        
        rng = np.random.RandomState(random_state)
        # 重みの初期化 (Xavier / He に準拠したスケーリング)
        self.W1 = rng.randn(n_in, n_hidden) / np.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, n_out) / np.sqrt(n_hidden)
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        """
        X: shape (N, D)
        戻り値:
          a1: (N, M) 隠れ層入力
          z1: (N, M) 隠れ層出力
          a2: (N, K) 出力層入力
          y:  (N, K) ネットワーク出力
        """
        a1 = X @ self.W1 + self.b1
        z1 = np.tanh(a1)
        a2 = z1 @ self.W2 + self.b2
        y = a2 # 線形出力
        return a1, z1, a2, y

    def predict(self, X):
        _, _, _, y = self.forward(X)
        return y

    def compute_loss_and_grads(self, X, T):
        """
        二乗和誤差 E = 0.5 * sum_n ||y_n - t_n||^2 + 0.5 * weight_decay * ||w||^2
        および解析的勾配の逆伝播計算
        """
        N = len(X)
        a1, z1, a2, y = self.forward(X)
        
        # 二乗和誤差
        data_loss = 0.5 * np.sum((y - T)**2)
        reg_loss = 0.5 * self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
        loss = data_loss + reg_loss
        
        # 誤差逆伝播 (Backpropagation)
        # 出力層の delta_k = y_k - t_k
        delta2 = (y - T) # (N, K)
        grad_W2 = z1.T @ delta2 + self.weight_decay * self.W2
        grad_b2 = np.sum(delta2, axis=0)
        
        # 隠れ層の delta_j = (1 - z_j^2) * sum_k W_jk delta_k
        delta1 = (delta2 @ self.W2.T) * (1.0 - z1**2) # (N, M)
        grad_W1 = X.T @ delta1 + self.weight_decay * self.W1
        grad_b1 = np.sum(delta1, axis=0)
        
        grads = {
            'W1': grad_W1, 'b1': grad_b1,
            'W2': grad_W2, 'b2': grad_b2
        }
        return loss, grads

    def fit(self, X, T, n_epochs=1000, lr=None, verbose=False):
        if lr is not None:
            self.lr = lr
        loss_history = []
        for epoch in range(n_epochs):
            loss, grads = self.compute_loss_and_grads(X, T)
            loss_history.append(loss)
            
            # 勾配降下更新
            self.W1 -= self.lr * grads['W1']
            self.b1 -= self.lr * grads['b1']
            self.W2 -= self.lr * grads['W2']
            self.b2 -= self.lr * grads['b2']
            
            if verbose and (epoch % (n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:4d}/{n_epochs}: Loss = {loss:.6f}")
        return loss_history

    def compute_jacobian(self, x):
        """
        単一サンプル x (shape: (D,)) に対するヤコビ行列 J_ki = dy_k / dx_i (PRML 5.3.4節)
        戻り値: shape (K, D)
        """
        a1 = x @ self.W1 + self.b1 # (M,)
        z1 = np.tanh(a1) # (M,)
        
        # dy_k / dx_i = sum_j W2_{j, k} * (1 - z1_j^2) * W1_{i, j}
        diag_deriv = (1.0 - z1**2) # (M,)
        # (W2.T @ diag(1-z^2)) @ W1.T -> shape (K, D)
        J = (self.W2.T * diag_deriv) @ self.W1.T
        return J


def gradient_check(model, X, T, eps=1e-5):
    """
    誤差逆伝播の勾配と数値微分（有限差分）の整合性を検証する (PRML 5.3.3節)
    """
    _, ana_grads = model.compute_loss_and_grads(X, T)
    
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = getattr(model, param_name)
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]
            
            param[idx] = orig_val + eps
            loss_plus, _ = model.compute_loss_and_grads(X, T)
            
            param[idx] = orig_val - eps
            loss_minus, _ = model.compute_loss_and_grads(X, T)
            
            param[idx] = orig_val
            num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            it.iternext()
            
        ana_grad = ana_grads[param_name]
        diff = np.linalg.norm(ana_grad - num_grad)
        denom = np.linalg.norm(ana_grad) + np.linalg.norm(num_grad) + 1e-12
        rel_diff = diff / denom
        assert rel_diff < 1e-4, f"Gradient check failed for {param_name}! rel_diff={rel_diff:.2e}"
    return True


class MixtureDensityNetwork:
    """
    混合密度ネットワーク (Mixture Density Network: MDN, PRML 5.6節)
    入力 x から、条件付き分布 p(t|x) = sum_k pi_k(x) N(t | mu_k(x), sigma_k^2(x)) の
    混合重み pi_k、平均 mu_k、分散 sigma_k^2 を予測する。
    """
    def __init__(self, n_in, n_hidden, n_components, random_state=42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.K = n_components # ガウス分布の数
        
        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(n_in, n_hidden) / np.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        
        # 隠れ層 -> 3つのパラメータグループ
        # 1. pi: K次元 (softmax)
        self.W_pi = rng.randn(n_hidden, self.K) / np.sqrt(n_hidden)
        self.b_pi = np.zeros(self.K)
        # 2. sigma: K次元 (exp)
        self.W_sig = rng.randn(n_hidden, self.K) / np.sqrt(n_hidden)
        self.b_sig = np.zeros(self.K)
        # 3. mu: K次元 (linear)
        self.W_mu = rng.randn(n_hidden, self.K) / np.sqrt(n_hidden)
        self.b_mu = rng.randn(self.K)

    def forward(self, X):
        a1 = X @ self.W1 + self.b1
        z = np.tanh(a1) # (N, n_hidden)
        
        # パラメータ出力 (PRML 式 5.145 - 5.148)
        a_pi = z @ self.W_pi + self.b_pi
        pi = softmax(a_pi, axis=-1)
        
        a_sig = z @ self.W_sig + self.b_sig
        sigma = np.exp(np.clip(a_sig, -10, 10)) # 正値制約
        
        a_mu = z @ self.W_mu + self.b_mu
        mu = a_mu # 線形
        return z, pi, sigma, mu, a_sig

    def compute_loss_and_grads(self, X, T):
        """
        負の対数尤度誤差 E = - sum_n ln [ sum_k pi_k N(t_n | mu_k, sigma_k^2) ]
        および各重みに対する解析的勾配 (PRML 式 5.153 - 5.157)
        """
        N = len(X)
        T = T.reshape(N, 1) # (N, 1)
        z, pi, sigma, mu, a_sig = self.forward(X)
        
        # 各成分の正規分布密度 N(t_n | mu_k, sigma_k^2)
        var = sigma**2 + 1e-12
        diff = T - mu # (N, K)
        # N(t | mu, sigma^2) = (2 pi sigma^2)^(-1/2) exp(- (t - mu)^2 / (2 sigma^2))
        norm_const = 1.0 / np.sqrt(2.0 * np.pi * var)
        exp_term = np.exp(- 0.5 * (diff**2) / var)
        comp_dens = norm_const * exp_term + 1e-30 # (N, K)
        
        # 結合確率密度 p(t_n | x_n) = sum_k pi_k N(...)
        joint = np.sum(pi * comp_dens, axis=1, keepdims=True) # (N, 1)
        nll = -np.sum(np.log(joint + 1e-30))
        
        # 責任度 (responsibilities) gamma_nk = pi_k N(...) / joint (PRML 式 5.154)
        gamma = (pi * comp_dens) / (joint + 1e-30) # (N, K)
        
        # 出力層のデルタ (PRML 式 5.155 - 5.157)
        delta_pi = pi - gamma                     # (N, K)
        delta_mu = gamma * (-diff / var)          # (N, K)
        delta_sig = gamma * (1.0 - (diff**2)/var) # (N, K)
        
        # 隠れ層への勾配
        grad_W_pi = z.T @ delta_pi; grad_b_pi = np.sum(delta_pi, axis=0)
        grad_W_mu = z.T @ delta_mu; grad_b_mu = np.sum(delta_mu, axis=0)
        grad_W_sig = z.T @ delta_sig; grad_b_sig = np.sum(delta_sig, axis=0)
        
        # 隠れ層入力へのデルタ
        delta_z = delta_pi @ self.W_pi.T + delta_mu @ self.W_mu.T + delta_sig @ self.W_sig.T
        delta_hidden = delta_z * (1.0 - z**2)
        grad_W1 = X.T @ delta_hidden; grad_b1 = np.sum(delta_hidden, axis=0)
        
        grads = {
            'W1': grad_W1, 'b1': grad_b1,
            'W_pi': grad_W_pi, 'b_pi': grad_b_pi,
            'W_mu': grad_W_mu, 'b_mu': grad_b_mu,
            'W_sig': grad_W_sig, 'b_sig': grad_b_sig
        }
        return nll, grads

    def fit(self, X, T, n_epochs=1000, lr=0.005, verbose=False):
        loss_history = []
        for epoch in range(n_epochs):
            loss, grads = self.compute_loss_and_grads(X, T)
            loss_history.append(loss)
            
            self.W1 -= lr * grads['W1']
            self.b1 -= lr * grads['b1']
            self.W_pi -= lr * grads['W_pi']
            self.b_pi -= lr * grads['b_pi']
            self.W_mu -= lr * grads['W_mu']
            self.b_mu -= lr * grads['b_mu']
            self.W_sig -= lr * grads['W_sig']
            self.b_sig -= lr * grads['b_sig']
            
            if verbose and (epoch % (n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:4d}/{n_epochs}: NLL = {loss:.4f}")
        return loss_history
