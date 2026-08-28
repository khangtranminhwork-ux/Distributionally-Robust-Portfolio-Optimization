import csv, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, NullFormatter

GLOBAL_COLORS = {'socp': '#000000', 'mv': '#d62728', 'mad': '#ff7f0e', 'moehle': '#1f77b4', 'dro_mad': '#1b9e77',
                 '1n': '#984ea3'}
NAME_MAP = {'socp': 'SOCP', 'mv': 'Classic MV', 'mad': 'Classic MAD', 'dro_mad': 'DRO-MAD', 'moehle': 'Moehle',
            '1n': '1/N'}
CATEGORIES = ["Data Processing", "Setup & Compilation", "Solver (Math)", "Post-processing"]
HATCHES = ['', '////', '....', 'xxxx']


def get_high_contrast_gradient(hex_color, step):
    rgb = np.array(mcolors.to_rgb(hex_color))
    fractions = [0.0, 0.45, 0.75, 0.92]
    return rgb + (np.array([1.0, 1.0, 1.0]) - rgb) * fractions[step]


def visualize_timing_gradient_log():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(SCRIPT_DIR, '..', 'data', 'run_time_analysis.csv')
    data = []

    # Check if file exists to prevent errors
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    with open(filename, mode='r') as f:
        for row in csv.DictReader(f):
            row_data = {"solver": row["solver"], "total": 0.0}
            for cat, short_cat in zip(
                    ["Data Processing", "Model Setup & Compilation", "Solver (Math)", "Backtest & Post-processing"],
                    CATEGORIES):
                # Ensure we don't have exactly 0 if we want it to show up on log scale,
                # but for stacked bars, 0 is fine as long as the 'bottom' eventually rises
                val = float(row.get(cat, 0.0))
                row_data[short_cat] = val
                row_data["total"] += val
            data.append(row_data)

    sorted_data = sorted(data, key=lambda x: x["total"])

    plt.figure(figsize=(16, 12))

    # Set the log scale BEFORE plotting to ensure the bar 'bottoms' are handled correctly
    plt.yscale('log')

    bottoms = np.zeros(len(sorted_data))

    for i, cat in enumerate(CATEGORIES):
        cat_values = [d[cat] for d in sorted_data]
        for s_idx, d in enumerate(sorted_data):
            face_color = get_high_contrast_gradient(GLOBAL_COLORS.get(d["solver"].lower(), "#95a5a6"), i)
            lum = 0.299 * face_color[0] + 0.587 * face_color[1] + 0.114 * face_color[2]

            # Draw the bar. Matplotlib handles log-scaling the heights automatically.
            plt.bar(s_idx, cat_values[s_idx], bottom=bottoms[s_idx], color=face_color,
                    edgecolor='white' if lum < 0.45 else 'black', linewidth=1.5, hatch=HATCHES[i], width=0.7)
        bottoms += np.array(cat_values)

    # --- FORMATTING ---
    plt.title('Time Breakdown (Log)', fontsize=40, fontweight='bold', pad=40)
    plt.ylabel('Seconds (Log Scale)', fontsize=32, fontweight='bold', labelpad=20)
    plt.xlabel('Optimization Models', fontsize=32, fontweight='bold', labelpad=20)

    # Set Y-axis limit: Log scale cannot start at 0.
    # Adjust 0.001 to 0.0001 if your data processing is extremely fast.
    plt.ylim(bottom=0.001)

    plt.xticks(range(len(sorted_data)), [NAME_MAP.get(d["solver"].lower(), d["solver"]) for d in sorted_data],
               fontsize=22)
    plt.yticks(fontsize=26)

    # Add minor gridlines for the log scale
    plt.grid(axis='y', which='major', linestyle='-', alpha=0.6, linewidth=2)
    plt.grid(axis='y', which='minor', linestyle='--', alpha=0.3, linewidth=1)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#cccccc', edgecolor='black', hatch=HATCHES[i], label=CATEGORIES[i])
        for i in range(len(CATEGORIES))
    ]
    plt.legend(handles=legend_elements[::-1], title="Components", title_fontsize='15', fontsize=12,
       loc='upper left', frameon=True, shadow=True, borderpad=1.5, labelspacing=1.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_timing_gradient_log()