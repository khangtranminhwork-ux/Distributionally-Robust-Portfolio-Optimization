import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from utils.models import run_all_models
import warnings

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, '..', 'data', '25_25_split.csv')
SAVE_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'spider_plots')
CACHE_FILE = os.path.join(SAVE_DIR, "backtest_results_cache.csv")

USE_CACHE = False

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

START_DATE = '2018-01-01'
END_DATE = '2025-12-31'
DELTAS = [0]
GAMMAS = [2]

GLOBAL_COLORS = {
    'SOCP': '#000000',
    'MV': '#d62728',
    'MAD': '#ff7f0e',
    'Moehle': '#1f77b4',
    'DROMAD': '#1b9e77',
    '1/N': '#984ea3'
}


# --- 2. DATA LOADING ---
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
        df = df.loc[mask].sort_values('Date')
        df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
        df_adj = df_adj.dropna(axis=1, how='any').dropna(how='any')
        return df_adj
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


# --- 3. EXECUTION WITH CACHING ---
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if USE_CACHE and os.path.exists(CACHE_FILE):
    print(f"--- Loading results from cache: {CACHE_FILE} ---")
    big_matrix = pd.read_csv(CACHE_FILE)
else:
    print("--- Starting Backtest Execution ---")
    data = load_data(CSV_FILE)
    print(data)
    all_results_list = []

    start_total = time.time()
    for delta in DELTAS:
        for gamma in GAMMAS:
            print(f"Running Delta: {delta}, Gamma: {gamma}...")
            batch_res = run_all_models(
                data=data,
                delta=delta,
                gamma=gamma,
                EST_WIN=252,
                PRED_WIN=21
            )
            all_results_list.append(batch_res)

    big_matrix = pd.concat(all_results_list, ignore_index=True)
    big_matrix.to_csv(CACHE_FILE, index=False)
    print(f"--- Backtest Complete. Saved to cache. Total Time: {time.time() - start_total:.2f}s ---")


# --- 4. PLOTTING LOGIC ---
def calculate_model_line_styles(df_all):
    styles = {}
    metrics = ['Mean Return', 'STD DEV', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
    models = df_all['Model'].unique()
    for model in models:
        subset = df_all[df_all['Model'] == model]
        delta_variance = subset.groupby('Gamma')[metrics].std().sum().sum()
        gamma_variance = subset.groupby('Delta')[metrics].std().sum().sum()

        if delta_variance < 1e-8 and gamma_variance > 1e-8:
            styles[model] = '--'
        elif gamma_variance < 1e-8 and delta_variance > 1e-8:
            styles[model] = ':'
        else:
            styles[model] = '-'
    return styles


def generate_spider_plots(df_all, target_path):
    line_styles = calculate_model_line_styles(df_all)
    metrics_map = {
        'Mean Return': 'Return',
        'STD DEV': 'Vol',
        'Sharpe Ratio': 'Sharpe',
        'Sortino Ratio': 'Sortino',
        'Max Drawdown': 'MDD'
    }
    display_metrics = list(metrics_map.values())
    num_vars = len(display_metrics)

    df_clean = df_all.dropna(subset=['Mean Return', 'STD DEV'])

    bounds = {}
    for col, display_name in metrics_map.items():
        if display_name in ['Vol', 'MDD']:
            bounds[display_name] = (df_clean[col].abs().min(), df_clean[col].abs().max())
        else:
            bounds[display_name] = (df_clean[col].min(), df_clean[col].max())

    def normalize(val, m_name):
        low, high = bounds[m_name]
        if high == low: return 0.5
        if m_name in ['Return', 'Sharpe', 'Sortino']:
            return (val - low) / (high - low)
        else:
            val = abs(val)
            return (high - val) / (high - low)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for d_val in sorted(df_clean['Delta'].unique()):
        for g_val in sorted(df_clean['Gamma'].unique()):
            subset = df_clean[(df_clean['Delta'] == d_val) & (df_clean['Gamma'] == g_val)]
            if subset.empty: continue

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            ax.set_rgrids([0.2, 0.4, 0.6, 0.8], labels=[], color="grey", size=7)

            for _, row in subset.iterrows():
                m_name = row['Model']

                if m_name == 'SOCP' and d_val == 0 and g_val == 0:
                    continue

                norm_vals = [
                    normalize(row['Mean Return'], 'Return'),
                    normalize(row['STD DEV'], 'Vol'),
                    normalize(row['Sharpe Ratio'], 'Sharpe'),
                    normalize(row['Sortino Ratio'], 'Sortino'),
                    normalize(row['Max Drawdown'], 'MDD')
                ]
                norm_vals = [np.clip(v, 0.01, 1) for v in norm_vals]
                values = norm_vals + [norm_vals[0]]

                color = GLOBAL_COLORS.get(m_name, '#CCCCCC')
                ax.plot(angles, values, color=color,
                        linewidth=3 if m_name == 'SOCP' else 2,
                        linestyle=line_styles.get(m_name, '-'),
                        label=m_name, alpha=0.9)
                ax.fill(angles, values, color=color, alpha=0.1)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), display_metrics, fontsize=11, fontweight='bold')

            ax.set_ylim(0, 1.1)
            ax.set_yticklabels([])

            plt.title(f"Risk-Reward Profile: Δ={d_val}, γ={g_val}", size=15, y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)

            fname = f"spider_D{str(d_val).replace('.', 'p')}_G{str(g_val).replace('.', 'p')}.png"
            plt.savefig(os.path.join(target_path, fname), dpi=300, bbox_inches='tight')
            plt.close()


# --- 5. EXECUTE PLOTTING ---
generate_spider_plots(big_matrix, SAVE_DIR)
print("Finished.")