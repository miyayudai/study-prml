import numpy as np

def rejection_sample(target_pdf, proposal_sampler, proposal_pdf, k, n_samples=1000, random_state=42):
    """
    棄却サンプリング (PRML 11.1.2節)
    k * proposal_pdf(z) >= target_pdf(z)
    """
    np.random.seed(random_state)
    accepted = []
    total_trials = 0
    
    while len(accepted) < n_samples:
        z_cand = proposal_sampler()
        u = np.random.uniform(0, k * proposal_pdf(z_cand))
        total_trials += 1
        if u <= target_pdf(z_cand):
            accepted.append(z_cand)
            
    return np.array(accepted), len(accepted) / total_trials


def metropolis_hastings(target_log_pdf, initial_state, proposal_sampler, proposal_log_pdf=None, n_samples=1000, random_state=42):
    """
    メトロポリス・ヘイスティングス法 (PRML 11.2節)
    対称提案分布の場合は標準メトロポリス法となる
    """
    np.random.seed(random_state)
    samples = [np.array(initial_state, dtype=float)]
    accepted_flags = [True]
    current = np.array(initial_state, dtype=float)
    
    for _ in range(n_samples - 1):
        candidate = proposal_sampler(current)
        
        # 受容比 alpha の計算 (対数領域)
        log_p_cand = target_log_pdf(candidate)
        log_p_curr = target_log_pdf(current)
        
        if proposal_log_pdf is not None:
            log_q_fwd = proposal_log_pdf(candidate, current) # q(cand | curr)
            log_q_rev = proposal_log_pdf(current, candidate) # q(curr | cand)
            log_alpha = log_p_cand + log_q_rev - log_p_curr - log_q_fwd
        else:
            log_alpha = log_p_cand - log_p_curr
            
        if np.log(np.random.uniform(0, 1)) < log_alpha:
            current = candidate
            accepted_flags.append(True)
        else:
            accepted_flags.append(False)
            
        samples.append(current.copy())
        
    return np.array(samples), np.array(accepted_flags)


def gibbs_sampler_2d(cond_sampler_x, cond_sampler_y, initial_state, n_samples=1000, random_state=42):
    """
    2変量ギブスサンプラー (PRML 11.3節, Figure 11.11)
    直交ステップ (x更新 -> y更新) の軌跡を記録
    """
    np.random.seed(random_state)
    trajectory = [np.array(initial_state, dtype=float)]
    x, y = initial_state
    
    for _ in range(n_samples):
        # 1. x を p(x | y) からサンプリング
        x = cond_sampler_x(y)
        trajectory.append(np.array([x, y]))
        # 2. y を p(y | x) からサンプリング
        y = cond_sampler_y(x)
        trajectory.append(np.array([x, y]))
        
    return np.array(trajectory)


def hamiltonian_monte_carlo(potential_energy, grad_potential, initial_state, n_samples=500, step_size=0.1, n_leapfrog=15, random_state=42):
    """
    ハイブリッド / ハミルトニアンモンテカルロ (HMC, PRML 11.5節)
    リープフロッグ積分器 (PRML Figure 11.14) によるシンプレクティックダイナミクス
    """
    np.random.seed(random_state)
    samples = [np.array(initial_state, dtype=float)]
    accepted_count = 0
    current_q = np.array(initial_state, dtype=float)
    dim = len(current_q)
    
    for _ in range(n_samples - 1):
        # 運動量 r ~ N(0, I) の生成
        current_r = np.random.normal(0, 1, size=dim)
        
        q = current_q.copy()
        r = current_r.copy()
        
        # リープフロッグ積分 (PRML 式 11.64 - 11.66)
        # 1. 半ステップ運動量更新
        r -= 0.5 * step_size * grad_potential(q)
        
        # 2. L-1 回の全ステップ更新
        for step in range(n_leapfrog):
            q += step_size * r
            if step != n_leapfrog - 1:
                r -= step_size * grad_potential(q)
                
        # 3. 最後の半ステップ運動量更新
        r -= 0.5 * step_size * grad_potential(q)
        
        # 運動量の符号反転 (時間反転対称性保証)
        r = -r
        
        # ハミルトニアンエネルギーの評価 H(q, r) = U(q) + K(r)
        current_U = potential_energy(current_q)
        current_K = 0.5 * np.sum(current_r**2)
        proposed_U = potential_energy(q)
        proposed_K = 0.5 * np.sum(r**2)
        
        dH = proposed_U + proposed_K - (current_U + current_K)
        
        # メトロポリス受容判定
        if np.log(np.random.uniform(0, 1)) < -dH:
            current_q = q
            accepted_count += 1
            
        samples.append(current_q.copy())
        
    return np.array(samples), accepted_count / (n_samples - 1)
