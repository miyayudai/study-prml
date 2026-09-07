"""
PRML Chapter 1.6: Information Theory
情報理論のコア関数とユーティリティ
- シャノンエントロピー (離散・2値): entropy_discrete, binary_entropy
- 微分エントロピー (ガウス分布解析解): differential_entropy_gaussian
- カルバック・ライブラー情報量 (離散・ガウス分布): kl_divergence_discrete, kl_divergence_gaussian
- 相互情報量 (2変量ガウス分布解析解): mutual_information_gaussian
- ハフマン符号化アルゴリズム: huffman_coding
"""

import heapq
import numpy as np


def entropy_discrete(p, base=2.0):
    """
    PRML 式 (1.104) 離散シャノンエントロピー
    H(p) = - \\sum_i p_i \\log_b(p_i)
    
    p: 確率分布ベクトル (和が1) または度数分布
    base: 対数の底 (2: bit, e: nat)
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative.")
    total = np.sum(p)
    if total <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    p = p / total
    # 0 log 0 = 0
    nonzero = p > 0
    p_nz = p[nonzero]
    if base == 2.0:
        return -np.sum(p_nz * np.log2(p_nz))
    elif base == np.e or base == 'e':
        return -np.sum(p_nz * np.log(p_nz))
    else:
        return -np.sum(p_nz * np.log(p_nz)) / np.log(base)


def binary_entropy(p, base=2.0):
    """
    PRML Figure 1.30: 2値エントロピー関数
    H(p) = -p \\log_b(p) - (1-p) \\log_b(1-p)
    """
    p = np.asarray(p, dtype=np.float64)
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1.0 - eps)
    if base == 2.0:
        log_fn = np.log2
    else:
        log_fn = lambda x: np.log(x) / np.log(base)
    return -(p_clipped * log_fn(p_clipped) + (1.0 - p_clipped) * log_fn(1.0 - p_clipped))


def differential_entropy_gaussian(cov):
    """
    PRML 式 (1.110) ガウス分布の微分エントロピー (単位: nat)
    H(x) = 1/2 \\ln {(2\\pi e)^D |\\Sigma|}
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        D = 1
        det_cov = float(cov)
    elif cov.ndim == 1:
        D = len(cov)
        det_cov = np.prod(cov)
    else:
        D = cov.shape[0]
        det_cov = np.linalg.det(cov)
    
    if det_cov <= 0:
        raise ValueError("Covariance matrix must be positive-definite.")
    
    return 0.5 * (D * np.log(2.0 * np.pi * np.e) + np.log(det_cov))


def kl_divergence_discrete(p, q, base=np.e):
    """
    PRML 式 (1.113) 離散KLダイバージェンス (相対エントロピー)
    KL(p || q) = - \\sum_i p_i \\ln (q_i / p_i) = \\sum_i p_i \\ln(p_i / q_i)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # p > 0 かつ q == 0 の場合は無限大
    if np.any((p > 0) & (q == 0)):
        return np.inf
    
    mask = p > 0
    ratio = p[mask] / q[mask]
    if base == np.e or base == 'e':
        return np.sum(p[mask] * np.log(ratio))
    elif base == 2.0:
        return np.sum(p[mask] * np.log2(ratio))
    else:
        return np.sum(p[mask] * np.log(ratio)) / np.log(base)


def kl_divergence_gaussian(mu1, cov1, mu2, cov2):
    """
    多変量ガウス分布 N(mu1, cov1) と N(mu2, cov2) の解析的 KL ダイバージェンス
    KL(N_1 || N_2) = 0.5 * [ tr(Sigma_2^{-1} Sigma_1) + (mu_2 - mu_1)^T Sigma_2^{-1} (mu_2 - mu_1) - D + ln(|Sigma_2| / |Sigma_1|) ]
    """
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    cov1 = np.atleast_2d(np.asarray(cov1, dtype=np.float64))
    cov2 = np.atleast_2d(np.asarray(cov2, dtype=np.float64))
    
    D = len(mu1)
    inv_cov2 = np.linalg.inv(cov2)
    diff = mu2 - mu1
    
    term_trace = np.trace(inv_cov2 @ cov1)
    term_quad = diff.T @ inv_cov2 @ diff
    term_det = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    
    return 0.5 * (term_trace + term_quad - D + term_det)


def mutual_information_gaussian(rho):
    """
    PRML 式 (1.121): 相関係数 rho をもつ2変量ガウス分布の相互情報量 (単位: nat)
    I(x, y) = - 1/2 \\ln (1 - \\rho^2)
    """
    rho = np.asarray(rho, dtype=np.float64)
    if np.any(np.abs(rho) >= 1.0):
        # rho = +-1 のときは無限大
        res = np.empty_like(rho)
        res[np.abs(rho) >= 1.0] = np.inf
        mask = np.abs(rho) < 1.0
        res[mask] = -0.5 * np.log(1.0 - rho[mask] ** 2)
        return res
    return -0.5 * np.log(1.0 - rho ** 2)


def huffman_coding(probabilities):
    """
    ハフマン符号化アルゴリズム
    probabilities: 辞書 {symbol: prob} またはリスト [prob_0, prob_1, ...]
    returns: (codes_dict, average_length)
    """
    if isinstance(probabilities, dict):
        items = [(p, [sym, ""]) for sym, p in probabilities.items()]
    else:
        items = [(p, [i, ""]) for i, p in enumerate(probabilities)]
    
    # 確率を正規化
    total = sum(item[0] for item in items)
    items = [(p / total, tree) for p, tree in items]
    
    heap = []
    for p, node in items:
        # (weight, tie_breaker, node)
        heapq.heappush(heap, (p, id(node), [node]))
        
    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        for pair in n1:
            pair[1] = '0' + pair[1]
        for pair in n2:
            pair[1] = '1' + pair[1]
        combined = n1 + n2
        heapq.heappush(heap, (p1 + p2, id(combined), combined))
        
    final_tree = heapq.heappop(heap)[2]
    codes = {sym: code for sym, code in final_tree}
    
    # 平均符号長
    if isinstance(probabilities, dict):
        avg_len = sum((probabilities[sym] / total) * len(codes[sym]) for sym in probabilities)
    else:
        avg_len = sum((probabilities[i] / total) * len(codes[i]) for i in range(len(probabilities)))
        
    return codes, avg_len
