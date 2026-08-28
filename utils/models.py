import pandas as pd
import numpy as np
import time
import cvxpy as cp
def optimize_dromad(returns, delta, rho, use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values
    xi = returns.values

    # Variables
    x = cp.Variable(n)
    y1 = cp.Variable(T)
    y2 = cp.Variable(T)
    t = cp.Variable()

    base_con = [cp.sum(x) == 1, x >= 0]

    # LP 1: Case where robust mean is above target
    c1 = base_con + [
        y1 >= mu @ x - xi @ x - delta,
        y1 >= -mu @ x + xi @ x + delta,
        y2 >= mu @ x - xi @ x + delta,
        y2 >= -mu @ x + xi @ x - delta,
        mu @ x - delta >= rho,
        t >= cp.sum(y1) / T + delta,
        t >= cp.sum(y2) / T + delta
    ]
    prob1 = cp.Problem(cp.Minimize(t), c1)

    # LP 2: Case where target return is within uncertainty bounds
    c2 = base_con + [
        y1 >= xi @ x - rho,
        y1 >= -xi @ x + rho,
        y2 >= mu @ x - xi @ x + delta,
        y2 >= -mu @ x + xi @ x - delta,
        mu @ x - delta <= rho,
        mu @ x + delta >= rho,
        t >= cp.sum(y1) / T + delta,
        t >= cp.sum(y2) / T + delta
    ]
    prob2 = cp.Problem(cp.Minimize(t), c2)

    best_obj = np.inf
    best_x = np.ones(n) / n
    final_status = "infeasible"

    solver_kwargs = {'solver': cp.ECOS}
    if use_custom_tol:
        solver_kwargs.update({
            'abstol': custom_tol,
            'reltol': custom_tol,
            'feastol': custom_tol
        })

    # Solve Both and find the best
    for p in [prob1, prob2]:
        try:
            p.solve(**solver_kwargs)
            if p.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                if p.value < best_obj:
                    best_obj = p.value
                    best_x = x.value
                    final_status = p.status
            elif final_status == "infeasible":
                final_status = p.status
        except:
            pass

    res = np.maximum(best_x, 0)
    return res / (res.sum() + 1e-8), final_status

def optimize_mad(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values
    w = cp.Variable(n)
    z = cp.Variable(T)

    mean_port = mu @ w
    constraints = [
        cp.sum(w) == 1, w >= 0,
        z >= (returns.values @ w) - mean_port,
        z >= -((returns.values @ w) - mean_port)
    ]

    mad_risk = cp.sum(z) / T
    obj = cp.Maximize(mu @ w - gamma * mad_risk)

    prob = cp.Problem(obj, constraints)

    # Prepare solver settings
    solver_opts = {'solver': cp.ECOS, 'verbose': False}
    if use_custom_tol:
        solver_opts.update({
            'abstol': custom_tol,
            'reltol': custom_tol,
            'feastol': custom_tol
        })

    try:
        prob.solve(**solver_opts)
        status = prob.status
    except Exception as e:
        status = f"Error: {str(e)}"

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status

def optimize_moehle(returns, prev_w, gamma, fixed_trd_cost,
                              lin_trd_cost, fixed_hld_cost, use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    w = cp.Variable(n)
    abs_trade = cp.Variable(n)
    z_trade = cp.Variable(n, boolean=True)
    z_hold = cp.Variable(n, boolean=True)

    risk_term = cp.quad_form(w, Sigma)
    cost_trade = fixed_trd_cost * cp.sum(z_trade) + lin_trd_cost * cp.sum(abs_trade)
    cost_hold = fixed_hld_cost * cp.sum(z_hold)

    obj = cp.Minimize(-mu @ w + gamma * risk_term + cost_trade + cost_hold)

    M = 1.0
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        abs_trade >= w - prev_w,
        abs_trade >= -(w - prev_w),
        abs_trade <= M * z_trade,
        w <= M * z_hold
    ]

    prob = cp.Problem(obj, constraints)

    solver_kwargs = {"verbose": False}
    if use_custom_tol:
        solver_kwargs.update({
            'abstol': custom_tol,
            'reltol': custom_tol,
            'feastol': custom_tol
        })

    status = "not_solved"
    try:
        prob.solve(solver=cp.ECOS_BB, **solver_kwargs)
        status = prob.status
    except cp.error.SolverError:
        try:
            prob.solve(**solver_kwargs)
            status = prob.status
        except Exception as e:
            status = f"Solver_Error: {str(e)}"

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status

def optimize_mv(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical stability
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    obj = cp.Maximize(mu @ w - gamma * risk)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, constraints)

    # Prepare solver options
    solver_opts = {'solver': cp.OSQP}
    if use_custom_tol:
        solver_opts.update({
            'eps_abs': custom_tol,
            'eps_rel': custom_tol,
            'eps_prim_inf': custom_tol,
            'eps_dual_inf': custom_tol
        })

    status = "error"
    try:
        prob.solve(**solver_opts)
        status = prob.status
    except Exception as e:
        status = f"Error: {str(e)}"

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status

def optimize_1n(returns):
    n = returns.shape[1]
    w = np.ones(n) / n
    status = "Analytical"
    return w, status

def optimize_socp(returns, prev_w, delta, gamma, G_TRD, G_HLD,
                  use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Regularization for numerical stability
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    try:
        L = np.linalg.cholesky(Sigma).T
    except:
        return np.ones(n) / n, "Cholesky_Fail"

    # Variables
    w = cp.Variable(n)
    t = cp.Variable()
    q = cp.Variable()
    turnover = cp.Variable(n)
    holding = cp.Variable(n)

    robust_penalty = delta

    obj = cp.Minimize(
        -mu @ w
        + robust_penalty * t
        + gamma * q
        + G_TRD * cp.sum(turnover)
        + G_HLD * cp.sum(holding)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.SOC(t, w),
        cp.sum_squares(L @ w) <= q,
        q >= 0,
        turnover >= w - prev_w,
        turnover >= -(w - prev_w),
        holding >= w
    ]

    prob = cp.Problem(obj, constraints)

    # Solver options based on toggle
    solver_opts = {'solver': cp.ECOS, 'verbose': False}
    if use_custom_tol:
        solver_opts.update({
            'abstol': custom_tol,
            'reltol': custom_tol,
            'feastol': custom_tol
        })

    try:
        prob.solve(**solver_opts)
        status = prob.status
    except Exception as e:
        status = f"Error: {str(e)}"

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status


def run_all_models(
        data=None,
        use_custom_tol=False,
        EST_WIN=252,
        PRED_WIN= 21,
        custom_tol=1e-7,
        delta=0.003,
        gamma=5,
        rho_target=0.0012,
        lin_trd_cost=0.0010,
        fix_trd_cost=0.0001,
        fix_hld_cost=0.0001,
        G_TRD = 0.001,
        G_HLD = 0.001
):
    """
    Runs DROMAD, MAD, Moehle, MV, 1/N, and SOCP models sequentially
    and returns a combined DataFrame of results.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    all_results = []

    # Define models to run
    # Format: (Name, Function Logic Block)
    model_names = ["DROMAD", "MAD", "Moehle", "MV", "1/N", "SOCP"]

    def calculate_stats(rets):
        s = pd.Series(rets)
        if s.empty: return [0] * 5
        mn = s.mean() * 252
        sd = s.std() * np.sqrt(252)
        sh = mn / sd if sd > 0 else 0
        ds = s[s < 0].std() * np.sqrt(252)
        so = mn / ds if ds > 0 else 0
        md = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
        return mn, sd, sh, so, md

    for model in model_names:
        print(f"Running model: {model}...")
        start_time = time.time()
        rets = []
        stats_summary = []
        prev_w = np.zeros(data.shape[1])

        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            if est.empty or pred.empty: continue

            # --- MODEL LOGIC SELECTION ---
            if model == "DROMAD":
                w, status = optimize_dromad(est, delta, rho_target, use_custom_tol, custom_tol)
                rets.extend(pred.values @ w)

            elif model == "MAD":
                w, status = optimize_mad(est, gamma/10, use_custom_tol, custom_tol)
                rets.extend(pred.values @ w)

            elif model == "Moehle":
                w, status = optimize_moehle(
                    est, prev_w, gamma, fix_trd_cost, lin_trd_cost, fix_hld_cost,
                    use_custom_tol=use_custom_tol, custom_tol=custom_tol
                )
                raw_ret = pred.values @ w
                trade_mag = np.abs(w - prev_w)
                cost_lin = lin_trd_cost * np.sum(trade_mag)
                epsilon = 1e-5
                cost_fix_trd = fix_trd_cost * np.sum((trade_mag > epsilon).astype(float))
                cost_fix_hld = fix_hld_cost * np.sum((w > epsilon).astype(float))
                raw_ret[0] -= (cost_lin + cost_fix_trd + cost_fix_hld)
                rets.extend(raw_ret)
                prev_w = w

            elif model == "MV":
                w, status = optimize_mv(est, gamma, use_custom_tol, custom_tol)
                rets.extend(pred.values @ w)

            elif model == "1/N":
                w, status = optimize_1n(est)
                rets.extend(pred.values @ w)

            elif model == "SOCP":
                w, status = optimize_socp(est, prev_w, delta, gamma, G_TRD, G_HLD,
                                          use_custom_tol, custom_tol)
                raw_ret = pred.values @ w
                trd_cost = G_TRD * np.sum(np.abs(w - prev_w))
                hld_cost = G_HLD * np.sum(np.abs(w))
                raw_ret[0] -= (trd_cost + hld_cost)
                rets.extend(raw_ret)
                prev_w = w

            stats_summary.append(status)

        # --- POST-BACKTEST CALCULATIONS ---
        runtime = time.time() - start_time
        mn, sd, sh, so, md = calculate_stats(rets)

        all_results.append({
            'Model': model,
            'Delta': delta,
            'Gamma': gamma,
            'Runtime': runtime,
            'Mean Return': mn,
            'STD DEV': sd,
            'Sharpe Ratio': sh,
            'Sortino Ratio': so,
            'Max Drawdown': md
        })

        # Solver health printout
        if __name__ == "__main__":
            unique_statuses = pd.Series(stats_summary).value_counts()
            print(f"--- {model} Status Summary ---")
            print(unique_statuses)

    return pd.DataFrame(all_results)
