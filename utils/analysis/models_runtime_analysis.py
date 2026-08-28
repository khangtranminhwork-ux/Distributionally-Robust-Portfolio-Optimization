import pandas as pd
import numpy as np
import time
import cvxpy as cp
import warnings
import csv
import os

warnings.filterwarnings('ignore')

# ---------------- GLOBAL TIMERS ----------------
timers = {
    "Data Processing": 0,
    "Model Setup & Compilation": 0,
    "Solver (Math)": 0,
    "Backtest & Post-processing": 0
}


def reset_timers():
    for key in timers:
        timers[key] = 0


# ---------------- SOLVER WRAPPER ----------------
def solve_with_timing(prob, solver_name, solver_kwargs):
    """Helper to split CVXPY overhead from pure solver time."""
    t_comp_start = time.time()

    # We call solve here
    try:
        t_before_call = time.time()
        prob.solve(**solver_kwargs)
        t_after_call = time.time()

        pure_math_time = 0
        if prob.solver_stats:
            pure_math_time = prob.solver_stats.solve_time if prob.solver_stats.solve_time else 0

        total_call_time = t_after_call - t_before_call
        overhead = max(0, total_call_time - pure_math_time)

        timers["Solver (Math)"] += pure_math_time
        # Compilation includes the time to build the problem + the CVXPY overhead during the call
        timers["Model Setup & Compilation"] += (t_before_call - t_comp_start) + overhead
        return prob.status
    except Exception as e:
        timers["Model Setup & Compilation"] += (time.time() - t_comp_start)
        return f"Error: {str(e)}"


# ---------------- OPTIMIZATION MODELS ----------------

def optimize_dromad(returns, delta, rho, use_custom_tol=False, custom_tol=1e-7):
    t_start = time.time()
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values
    xi = returns.values

    x = cp.Variable(n)
    y1, y2 = cp.Variable(T), cp.Variable(T)
    t = cp.Variable()

    base_con = [cp.sum(x) == 1, x >= 0]
    c1 = base_con + [
        y1 >= mu @ x - xi @ x - delta, y1 >= -mu @ x + xi @ x + delta,
        y2 >= mu @ x - xi @ x + delta, y2 >= -mu @ x + xi @ x - delta,
        mu @ x - delta >= rho, t >= cp.sum(y1) / T + delta, t >= cp.sum(y2) / T + delta
    ]
    prob1 = cp.Problem(cp.Minimize(t), c1)

    c2 = base_con + [
        y1 >= xi @ x - rho, y1 >= -xi @ x + rho,
        y2 >= mu @ x - xi @ x + delta, y2 >= -mu @ x + xi @ x - delta,
        mu @ x - delta <= rho, mu @ x + delta >= rho,
        t >= cp.sum(y1) / T + delta, t >= cp.sum(y2) / T + delta
    ]
    prob2 = cp.Problem(cp.Minimize(t), c2)

    timers["Model Setup & Compilation"] += (time.time() - t_start)

    solver_kwargs = {'solver': cp.ECOS}
    if use_custom_tol:
        solver_kwargs.update({'abstol': custom_tol, 'reltol': custom_tol, 'feastol': custom_tol})

    best_obj, best_x, final_status = np.inf, np.ones(n) / n, "infeasible"

    # Solve 1
    solve_with_timing(prob1, 'ECOS', solver_kwargs)
    if prob1.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        best_obj, best_x, final_status = prob1.value, x.value, prob1.status

    # Solve 2
    solve_with_timing(prob2, 'ECOS', solver_kwargs)
    if prob2.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and prob2.value < best_obj:
        best_x, final_status = x.value, prob2.status

    t_post = time.time()
    res = np.maximum(best_x, 0)
    weights = res / (res.sum() + 1e-8)
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return weights, final_status


def optimize_mad(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    t_start = time.time()
    n, T = returns.shape[1], returns.shape[0]
    mu = returns.mean().values
    w, z = cp.Variable(n), cp.Variable(T)
    constraints = [cp.sum(w) == 1, w >= 0, z >= (returns.values @ w) - (mu @ w),
                   z >= -((returns.values @ w) - (mu @ w))]
    prob = cp.Problem(cp.Maximize(mu @ w - gamma * (cp.sum(z) / T)), constraints)
    timers["Model Setup & Compilation"] += (time.time() - t_start)

    solver_opts = {'solver': cp.ECOS}
    if use_custom_tol:
        solver_opts.update({'abstol': custom_tol, 'reltol': custom_tol, 'feastol': custom_tol})

    status = solve_with_timing(prob, 'ECOS', solver_opts)

    t_post = time.time()
    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        res_w = np.ones(n) / n
    else:
        res = np.maximum(w.value, 0)
        res_w = res / res.sum()
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_w, status


def optimize_moehle(returns, prev_w, gamma, fixed_trd_cost, lin_trd_cost, fixed_hld_cost, use_custom_tol=False,
                    custom_tol=1e-7):
    t_start = time.time()
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values + np.eye(n) * 1e-8
    w, abs_trade = cp.Variable(n), cp.Variable(n)
    z_trade, z_hold = cp.Variable(n, boolean=True), cp.Variable(n, boolean=True)

    obj = cp.Minimize(-mu @ w + gamma * cp.quad_form(w, Sigma) +
                      fixed_trd_cost * cp.sum(z_trade) + lin_trd_cost * cp.sum(abs_trade) + fixed_hld_cost * cp.sum(
        z_hold))
    constraints = [cp.sum(w) == 1, w >= 0, abs_trade >= w - prev_w, abs_trade >= -(w - prev_w),
                   abs_trade <= 1.0 * z_trade, w <= 1.0 * z_hold]
    prob = cp.Problem(obj, constraints)
    timers["Model Setup & Compilation"] += (time.time() - t_start)

    status = solve_with_timing(prob, 'ECOS_BB', {})

    t_post = time.time()
    res_w = np.ones(n) / n if w.value is None else np.maximum(w.value, 0) / np.maximum(w.value, 0).sum()
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_w, status


def optimize_mv(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    t_start = time.time()
    n = returns.shape[1]
    mu, Sigma = returns.mean().values, returns.cov().values + np.eye(n) * 1e-8
    w = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma)), [cp.sum(w) == 1, w >= 0])
    timers["Model Setup & Compilation"] += (time.time() - t_start)

    solver_opts = {'solver': cp.OSQP}
    if use_custom_tol:
        solver_opts.update({'eps_abs': custom_tol, 'eps_rel': custom_tol})

    status = solve_with_timing(prob, 'OSQP', solver_opts)

    t_post = time.time()
    res_w = np.ones(n) / n if w.value is None else np.maximum(w.value, 0) / np.maximum(w.value, 0).sum()
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_w, status


def optimize_1n(returns):
    t_post = time.time()
    n = returns.shape[1]
    w = np.ones(n) / n
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return w, "Analytical"


def optimize_socp(returns, prev_w, delta, gamma, G_TRD, G_HLD, use_custom_tol=False, custom_tol=1e-7):
    t_start = time.time()
    n = returns.shape[1]
    mu, Sigma = returns.mean().values, returns.cov().values + np.eye(n) * 1e-8
    L = np.linalg.cholesky(Sigma).T
    w, t, q = cp.Variable(n), cp.Variable(), cp.Variable()
    turnover, holding = cp.Variable(n), cp.Variable(n)

    obj = cp.Minimize(-mu @ w + delta * t + gamma * q + G_TRD * cp.sum(turnover) + G_HLD * cp.sum(holding))
    constraints = [cp.sum(w) == 1, w >= 0, cp.SOC(t, w), cp.sum_squares(L @ w) <= q, q >= 0,
                   turnover >= w - prev_w, turnover >= -(w - prev_w), holding >= w]
    prob = cp.Problem(obj, constraints)
    timers["Model Setup & Compilation"] += (time.time() - t_start)

    solver_opts = {'solver': cp.ECOS}
    if use_custom_tol:
        solver_opts.update({'abstol': custom_tol, 'reltol': custom_tol})

    status = solve_with_timing(prob, 'ECOS', solver_opts)

    t_post = time.time()
    res_w = np.ones(n) / n if w.value is None else np.maximum(w.value, 0) / np.maximum(w.value, 0).sum()
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_w, status


# ---------------- DATA & LOGGING ----------------

def update_analyzed_csv(solver_name, timers_dict, filename="analyzed_results.csv"):
    fieldnames = ["solver", "Data Processing", "Model Setup & Compilation", "Solver (Math)",
                  "Backtest & Post-processing"]
    file_exists = os.path.isfile(filename)

    rows = []
    if file_exists:
        with open(filename, 'r') as f:
            rows = list(csv.DictReader(f))

    found = False
    for row in rows:
        if row["solver"] == solver_name:
            row.update({k: f"{v:.6f}" for k, v in timers_dict.items()})
            found = True
            break
    if not found:
        new_row = {"solver": solver_name}
        new_row.update({k: f"{v:.6f}" for k, v in timers_dict.items()})
        rows.append(new_row)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_all_models_with_timing(data, EST_WIN=252, PRED_WIN=21):
    model_names = ["DROMAD", "MAD", "Moehle", "MV", "1/N", "SOCP"]
    final_report = []

    for model in model_names:
        print(f"Starting {model}...")
        reset_timers()
        rets = []
        prev_w = np.zeros(data.shape[1])

        # Hyperparams
        delta, gamma, rho_target = 0.003, 0.5, 0.0012
        lin_trd, fix_trd, fix_hld = 0.001, 0.0001, 0.0001

        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            # Data Processing Timer
            t_slice = time.time()
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_slice)

            if est.empty or pred.empty: continue

            # Core Logic
            if model == "DROMAD":
                w, _ = optimize_dromad(est, delta, rho_target)
            elif model == "MAD":
                w, _ = optimize_mad(est, gamma)
            elif model == "Moehle":
                w, _ = optimize_moehle(est, prev_w, gamma, fix_trd, lin_trd, fix_hld)
            elif model == "MV":
                w, _ = optimize_mv(est, gamma)
            elif model == "1/N":
                w, _ = optimize_1n(est)
            elif model == "SOCP":
                w, _ = optimize_socp(est, prev_w, delta, gamma, 0.001, 0.001)

            # Backtest Timing
            t_back = time.time()
            period_rets = pred.values @ w
            # Handle costs for trade-sensitive models
            if model in ["Moehle", "SOCP"]:
                cost = 0.001 * np.sum(np.abs(w - prev_w))
                period_rets[0] -= cost
            rets.extend(period_rets)
            prev_w = w
            timers["Backtest & Post-processing"] += (time.time() - t_back)

        # Final Stats Timing
        t_final = time.time()
        s = pd.Series(rets)
        mean_ret = s.mean() * 252 if not s.empty else 0
        timers["Backtest & Post-processing"] += (time.time() - t_final)

        # Save to CSV
        update_analyzed_csv(model, timers)
        final_report.append({"Model": model, "Total Time": sum(timers.values()), "Ann. Ret": mean_ret})

    return pd.DataFrame(final_report)
