import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# ----------------------------
# 1. Leer datasets
# ----------------------------
df_biased = pd.read_csv("synthetic_fairness_data_biased.csv")
df_neutral = pd.read_csv("synthetic_fairness_data_neutral.csv")

sns.set(style="whitegrid")

# ----------------------------
# 2. MÉTRICAS EXTENDIDAS
# ----------------------------
def compute_phi_ind(df, group_col):
    p_group = df[group_col].value_counts(normalize=True).sort_index()
    alert_subset = df[df['alert'] == 1]
    p_group_in_alerts = alert_subset[group_col].value_counts(normalize=True).sort_index()
    phi_ind = (p_group_in_alerts / p_group).fillna(0)
    return phi_ind

def f1_per_group(df, group_col):
    return {g: f1_score(sub['confirmed'], sub['alert'], zero_division=0)
            for g, sub in df.groupby(group_col)}

phi_ind_biased = compute_phi_ind(df_biased, 'platform')
phi_ind_neutral = compute_phi_ind(df_neutral, 'platform')
f1_biased = f1_per_group(df_biased, 'platform')
f1_neutral = f1_per_group(df_neutral, 'platform')

# Calibration Sufficiency (extendida)
suff_biased = df_biased.groupby(['score', 'platform'])['confirmed'].mean().unstack()
suff_neutral = df_neutral.groupby(['score', 'platform'])['confirmed'].mean().unstack()
delta_biased = suff_biased.max(axis=1) - suff_biased.min(axis=1)
delta_neutral = suff_neutral.max(axis=1) - suff_neutral.min(axis=1)

# ----------------------------
# 3. Formatear métricas extendidas
# ----------------------------
indep_table = pd.DataFrame([
    [r'$\phi_{\text{ind}}$', 'Extended', group, phi_ind_biased[group], phi_ind_neutral[group]]
    for group in phi_ind_biased.index
], columns=['Dimension', 'Metric', 'Group', 'Biased', 'Neutral'])

sep_table = pd.DataFrame([
    [r'$\phi_{\text{sep}}$', 'Extended', group, f1_biased[group], f1_neutral[group]]
    for group in f1_biased
], columns=['Dimension', 'Metric', 'Group', 'Biased', 'Neutral'])

suff_table = pd.DataFrame([
    [r'$\delta_{\text{cal}}$', 'Extended', f"Score = {score}", delta_biased[score], delta_neutral[score]]
    for score in delta_biased.index
], columns=['Dimension', 'Metric', 'Group', 'Biased', 'Neutral'])

# ----------------------------
# 4. MÉTRICAS CLÁSICAS
# ----------------------------
def classical_independence(df, group_col='platform'):
    return df.groupby(group_col)['alert'].mean()

def classical_separation(df, group_col='platform'):
    return df[df['confirmed'] == 1].groupby(group_col)['alert'].mean()

def classical_sufficiency(df, group_col='platform'):
    return df[df['alert'] == 1].groupby(group_col)['confirmed'].mean()

classic_rows = []

# Independence
ci_biased = classical_independence(df_biased)
ci_neutral = classical_independence(df_neutral)
for group in ci_biased.index:
    classic_rows.append([r'$\phi_{\text{ind}}$', 'Classical', group, ci_biased[group], ci_neutral[group]])

# Separation
cs_biased = classical_separation(df_biased)
cs_neutral = classical_separation(df_neutral)
for group in cs_biased.index:
    classic_rows.append([r'$\phi_{\text{sep}}$', 'Classical', group, cs_biased[group], cs_neutral[group]])

# Sufficiency (clásica)
cal_biased = classical_sufficiency(df_biased)
cal_neutral = classical_sufficiency(df_neutral)
for group in cal_biased.index:
    classic_rows.append([r'$\delta_{\text{cal}}$', 'Classical', group, cal_biased[group], cal_neutral[group]])

classic_table = pd.DataFrame(classic_rows, columns=['Dimension', 'Metric', 'Group', 'Biased', 'Neutral'])

# ----------------------------
# 5. Exportar tabla completa
# ----------------------------
combined_table = pd.concat([indep_table, sep_table, suff_table, classic_table], ignore_index=True)
combined_table.to_csv("metric_table_all.csv", index=False)

# ----------------------------
# 6. Funciones de visualización
# ----------------------------
def plot_comparison(df, title, filename):
    df = df.copy()
    df['Biased'] = pd.to_numeric(df['Biased'], errors='coerce')
    df['Neutral'] = pd.to_numeric(df['Neutral'], errors='coerce')
    plot_df = df.melt(id_vars=['Dimension', 'Metric', 'Group'],
                      value_vars=['Biased', 'Neutral'],
                      var_name='Dataset', value_name='Value')
    plot_df.dropna(subset=['Value'], inplace=True)
    plot_df['Label'] = plot_df['Metric'] + ', ' + plot_df['Dataset']
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x='Group', y='Value', hue='Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_comparison_per_metric(df, dimension_label, title, filename):
    subset = df[df['Dimension'] == dimension_label].copy()
    subset['Biased'] = pd.to_numeric(subset['Biased'], errors='coerce')
    subset['Neutral'] = pd.to_numeric(subset['Neutral'], errors='coerce')
    plot_df = subset.melt(id_vars=['Metric', 'Group'],
                          value_vars=['Biased', 'Neutral'],
                          var_name='Dataset', value_name='Value')
    plot_df.dropna(subset=['Value'], inplace=True)
    plot_df['Label'] = plot_df['Metric'] + ', ' + plot_df['Dataset']
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x='Group', y='Value', hue='Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ----------------------------
# 7. Gráficos Extendidas: Biased vs Neutral
# ----------------------------
plot_comparison(indep_table, "Operational Independence: Biased vs Neutral", "phi_ind_comparison_full.png")
plot_comparison(sep_table, "Detection Separation: Biased vs Neutral", "phi_sep_comparison_full.png")
plot_comparison(suff_table, "Calibration Sufficiency: Biased vs Neutral", "delta_cal_comparison_full.png")

# ----------------------------
# 8. Comparación Clásicas vs Extendidas: NEUTRAL
# ----------------------------
for dim, title, file in [
    (r'$\phi_{\text{ind}}$', "Neutral Dataset: Independence", "comparison_neutral_phi_ind.png"),
    (r'$\phi_{\text{sep}}$', "Neutral Dataset: Separation", "comparison_neutral_phi_sep.png"),
]:
    plot_comparison_per_metric(combined_table.copy(), dim, title, file)

# ----------------------------
# 9. Comparación Clásicas vs Extendidas: BIASED
# ----------------------------
for dim, title, file in [
    (r'$\phi_{\text{ind}}$', "Biased Dataset: Independence", "comparison_biased_phi_ind.png"),
    (r'$\phi_{\text{sep}}$', "Biased Dataset: Separation", "comparison_biased_phi_sep.png"),
]:
    plot_comparison_per_metric(combined_table.copy(), dim, title, file)

# ----------------------------
# 10. Gráficos SEPARADOS para Sufficiency clásica y extendida
# ----------------------------

# Sufficienty extendida: por score
plot_comparison(suff_table,
                "Calibration Sufficiency (Extended) by Score",
                "sufficiency_extended_delta_cal.png")

# Sufficienty clásica: por plataforma
suff_classic_df = classic_table[classic_table['Dimension'] == r'$\delta_{\text{cal}}$'].copy()
suff_classic_df['Biased'] = pd.to_numeric(suff_classic_df['Biased'], errors='coerce')
suff_classic_df['Neutral'] = pd.to_numeric(suff_classic_df['Neutral'], errors='coerce')

plot_df = suff_classic_df.melt(id_vars=['Metric', 'Group'],
                               value_vars=['Biased', 'Neutral'],
                               var_name='Dataset', value_name='Value')
plot_df.dropna(subset=['Value'], inplace=True)
plot_df['Label'] = plot_df['Metric'] + ', ' + plot_df['Dataset']

plt.figure(figsize=(8, 5))
sns.barplot(data=plot_df, x='Group', y='Value', hue='Label')
plt.title("Calibration Sufficiency (Classical) by Platform")
plt.tight_layout()
plt.savefig("sufficiency_classical_platform.png")
plt.close()
