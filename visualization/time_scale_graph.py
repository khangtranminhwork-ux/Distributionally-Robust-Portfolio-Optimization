import pandas as pd
import matplotlib.pyplot as plt
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'run_time_scale_analysis.csv')
GLOBAL_COLORS = {'SOCP': '#000000', 'Classic MV': '#d62728', 'Classic MAD': '#ff7f0e', 'Moehle_Sampled': '#1f77b4',
                 'DRO-MAD': '#1b9e77', '1/N': '#984ea3'}
NAME_MAP = {'socp': 'SOCP', 'mv': 'Classic MV', 'mad': 'Classic MAD', 'dro_mad': 'DRO-MAD', 'moehle': 'Moehle_Sampled',
            '1n': '1/N'}


def load_and_plot_scaling_log():
    if not os.path.exists(CSV_FILE):
        print(f"File not found: {CSV_FILE}")
        return

    df_scale = pd.read_csv(CSV_FILE)
    data_cols = sorted([c for c in df_scale.columns if c.startswith('N=')], key=lambda x: int(x.split('=')[1]))

    # --- INCREASED HEIGHT ---
    plt.figure(figsize=(16, 20))

    plot_data_list = []

    for _, row in df_scale.iterrows():
        raw_name = str(row['solver']).strip().lower()
        model_name = NAME_MAP.get(raw_name, row['solver'])

        x_vals, y_vals = [], []
        for col in data_cols:
            val = str(row[col])
            if pd.isna(row[col]) or val.strip() == "" or val.lower() == 'nan':
                continue

            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
            if numbers:
                total_time = sum(float(n) for n in numbers)
                if total_time > 0:
                    x_vals.append(int(col.split('=')[1]))
                    y_vals.append(total_time)

        if x_vals:
            # Sort by runtime at N=100 (or the max available X)
            val_at_ref = 0
            for x, y in zip(x_vals, y_vals):
                if x == 100:
                    val_at_ref = y
                    break

            plot_data_list.append({
                'name': model_name,
                'x': x_vals,
                'y': y_vals,
                'sort_val': val_at_ref,
                'color': GLOBAL_COLORS.get(model_name, '#95a5a6'),
                'zorder': 10 if model_name == 'SOCP' else 5
            })

    plot_data_list.sort(key=lambda item: item['sort_val'], reverse=True)

    for item in plot_data_list:
        plt.plot(item['x'], item['y'], label=item['name'], color=item['color'],
                 marker='o', linewidth=6, markersize=16, zorder=item['zorder'])

    # --- ONLY Y-AXIS LOG SCALE ---
    # plt.xscale('log') # Removed this line
    plt.yscale('log')

    # --- STYLING ---
    plt.title('Computational Scalability (Log)', fontsize=40, fontweight='bold', pad=40)
    plt.xlabel('Number of Assets ($N$)', fontsize=32, fontweight='bold', labelpad=20)
    plt.ylabel('Total Time [Seconds]', fontsize=32, fontweight='bold', labelpad=20)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    plt.grid(False)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    load_and_plot_scaling_log()