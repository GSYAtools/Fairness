import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# ----------------------------
# Cargar archivos UNSW-NB15
# ----------------------------
data_dir = 'C:/Fairness/UNSW'
usecols = [4, 7, 17, 48]  # proto, dur, Spkts, Label
column_names = ['proto', 'dur', 'Spkts', 'Label']

dfs = []
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path, header=None, usecols=usecols, names=column_names, low_memory=False)
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# ----------------------------
# Preprocesamiento
# ----------------------------
df['proto'] = df['proto'].astype(str)
df['dur'] = pd.to_numeric(df['dur'], errors='coerce')
df['Spkts'] = pd.to_numeric(df['Spkts'], errors='coerce')
df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
df = df.dropna()

# Definir platform
top_protocols = df['proto'].value_counts().nlargest(5).index
df['platform'] = df['proto'].apply(lambda p: p if p in top_protocols else 'Others')

# Variables de fairness
combined = np.log1p(df['dur'].clip(lower=0)) + np.log1p(df['Spkts'].clip(lower=1))
df['alert'] = (combined > combined.median()).astype(int)
df['confirmed'] = df['Label'].astype(int)
df['score'] = pd.qcut(combined, 5, labels=[1, 2, 3, 4, 5])

# ----------------------------
# Calcular métricas por grupo
# ----------------------------

# φ_ind: ratio alertas vs prevalencia
p_total = df['platform'].value_counts(normalize=True)
p_alert = df[df['alert'] == 1]['platform'].value_counts(normalize=True)
phi_ind = (p_alert / p_total).fillna(0)

# φ_sep: F1 por grupo
phi_sep = df.groupby('platform').apply(lambda g: f1_score(g['confirmed'], g['alert'], zero_division=0))

# δ_cal: promedio de desviación por score
score_group = df.groupby(['score', 'platform'])['confirmed'].mean().unstack()
delta_cal = score_group.apply(lambda col: (col - col.mean()).abs().mean())

# ----------------------------
# Construir tabla
# ----------------------------
df_metrics = pd.DataFrame({
    'phi_ind': phi_ind.round(4),
    'phi_sep': phi_sep.round(4),
    'delta_cal': delta_cal.round(4)
}).fillna(0)

# Activación empírica
df_metrics['act_phi_ind'] = df_metrics['phi_ind'] > 1.2
df_metrics['act_phi_sep'] = df_metrics['phi_sep'] < 0.1
df_metrics['act_delta_cal'] = df_metrics['delta_cal'] > 0.15
df_metrics['activated'] = df_metrics[['act_phi_ind', 'act_phi_sep', 'act_delta_cal']].sum(axis=1)

# Correlaciones
correlation = df_metrics[['phi_ind', 'phi_sep', 'delta_cal']].corr()

print("\n=== Métricas por grupo ===")
print(df_metrics[['phi_ind', 'phi_sep', 'delta_cal', 'activated']])

print("\n=== Correlaciones ===")
print(correlation)

df_metrics.to_csv("fairness_metric_compatibility.csv")

# Tabla LaTeX
latex_lines = [
    "\\begin{table}[ht]",
    "\\centering",
    "\\caption{Joint behavior of fairness metrics per platform.}",
    "\\label{tab:compatibility_joint}",
    "\\begin{tabular}{lccc}",
    "\\hline",
    "\\textbf{Platform} & $\\phi_{\\text{ind}}$ & $\\phi_{\\text{sep}}$ & $\\delta_{\\text{cal}}$ \\\\",
    "\\hline"
]

for idx, row in df_metrics.iterrows():
    latex_lines.append(f"{idx} & {row['phi_ind']:.4f} & {row['phi_sep']:.4f} & {row['delta_cal']:.4f} \\\\")

latex_lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

with open("fairness_metric_compatibility.tex", "w") as f:
    f.write("\n".join(latex_lines))

print("\nLaTeX guardado como 'fairness_metric_compatibility.tex'")