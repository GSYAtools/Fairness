import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

# ----------------------------
# 1. Load all datasets
# ----------------------------
datasets = {
    'Neutral': pd.read_csv("synthetic_fairness_data_neutral.csv"),
    'Biased (full)': pd.read_csv("synthetic_fairness_data_biased.csv"),
    'Alert-only bias': pd.read_csv("synthetic_fairness_data_alert_only.csv"),
    'Calibration-only bias': pd.read_csv("synthetic_fairness_data_calibration_only.csv"),
}

# ----------------------------
# 2. Metric functions
# ----------------------------
def compute_phi_ind(df, group_col='platform'):
    p_group = df[group_col].value_counts(normalize=True).sort_index()
    p_group_in_alerts = df[df['alert'] == 1][group_col].value_counts(normalize=True).sort_index()
    return (p_group_in_alerts / p_group).fillna(0)

def f1_per_group(df, group_col='platform'):
    return {g: f1_score(sub['confirmed'], sub['alert'], zero_division=0)
            for g, sub in df.groupby(group_col)}

def compute_delta_cal(df, group_col='platform'):
    cal = df.groupby(['score', group_col])['confirmed'].mean().unstack()
    return cal.max(axis=1) - cal.min(axis=1)

# ----------------------------
# 3. Compute metrics for all scenarios
# ----------------------------
all_results = []

for scenario_name, df in datasets.items():
    phi_ind = compute_phi_ind(df)
    phi_sep = f1_per_group(df)
    delta_cal = compute_delta_cal(df)

    for platform in phi_ind.index:
        all_results.append({
            'Scenario': scenario_name,
            'Metric': 'phi_ind',
            'Group': platform,
            'Value': phi_ind[platform]
        })

    for platform, f1_val in phi_sep.items():
        all_results.append({
            'Scenario': scenario_name,
            'Metric': 'phi_sep',
            'Group': platform,
            'Value': f1_val
        })

    for score_level in delta_cal.index:
        all_results.append({
            'Scenario': scenario_name,
            'Metric': 'delta_cal',
            'Group': f'Score {score_level}',
            'Value': delta_cal[score_level]
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv("metric_table_all_scenarios.csv", index=False)
print("Saved: metric_table_all_scenarios.csv")

# ----------------------------
# 4. Generate comparison plots
# ----------------------------

# Plot phi_ind across all scenarios
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for idx, (scenario_name, df) in enumerate(datasets.items()):
    phi_ind = compute_phi_ind(df)
    axes[idx].bar(phi_ind.index, phi_ind.values, color=['#4C72B0', '#55A868', '#C44E52'])
    axes[idx].set_title(scenario_name, fontsize=11)
    axes[idx].set_ylabel(r'$\phi_{\mathrm{ind}}$' if idx == 0 else '')
    axes[idx].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate(phi_ind.values):
        axes[idx].text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=9)
fig.suptitle(r'Operational Independence ($\phi_{\mathrm{ind}}$) across scenarios', fontsize=13)
plt.tight_layout()
plt.savefig("phi_ind_all_scenarios.png")
plt.close()
print("Saved: phi_ind_all_scenarios.png")

# Plot phi_sep across all scenarios
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for idx, (scenario_name, df) in enumerate(datasets.items()):
    phi_sep = f1_per_group(df)
    platforms = sorted(phi_sep.keys())
    values = [phi_sep[p] for p in platforms]
    axes[idx].bar(platforms, values, color=['#4C72B0', '#55A868', '#C44E52'])
    axes[idx].set_title(scenario_name, fontsize=11)
    axes[idx].set_ylabel(r'$\phi_{\mathrm{sep}}$ (F1)' if idx == 0 else '')
    for i, v in enumerate(values):
        axes[idx].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)
fig.suptitle(r'Detection Separation ($\phi_{\mathrm{sep}}$) across scenarios', fontsize=13)
plt.tight_layout()
plt.savefig("phi_sep_all_scenarios.png")
plt.close()
print("Saved: phi_sep_all_scenarios.png")

# Plot delta_cal across all scenarios
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for idx, (scenario_name, df) in enumerate(datasets.items()):
    delta_cal = compute_delta_cal(df)
    axes[idx].plot(delta_cal.index, delta_cal.values, marker='o', color='#8172B2')
    axes[idx].set_title(scenario_name, fontsize=11)
    axes[idx].set_ylabel(r'$\delta_{\mathrm{cal}}$' if idx == 0 else '')
    axes[idx].set_xlabel('Score')
    for i, v in zip(delta_cal.index, delta_cal.values):
        axes[idx].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
fig.suptitle(r'Calibration Sufficiency ($\delta_{\mathrm{cal}}$) across scenarios', fontsize=13)
plt.tight_layout()
plt.savefig("delta_cal_all_scenarios.png")
plt.close()
print("Saved: delta_cal_all_scenarios.png")

# ----------------------------
# 5. Print summary table
# ----------------------------
print("\n" + "=" * 80)
print("METRIC ISOLATION ANALYSIS")
print("=" * 80)

pivot = results_df.pivot_table(index=['Metric', 'Group'], columns='Scenario', values='Value')
print(pivot.to_string())
print("\nExpected behavior:")
print("- Alert-only bias: phi_ind should deviate, phi_sep and delta_cal should remain neutral-like")
print("- Calibration-only bias: delta_cal should deviate, phi_ind and phi_sep should remain neutral-like")
print("- Full bias: All metrics should deviate")
print("- Neutral: All metrics should be near ideal values")
