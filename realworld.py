import os
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
# Carga de datos
# ----------------------------
data_dir = 'C:/Fairness/UNSW'
usecols = [4, 7, 17, 29, 48]  # proto, dur, Spkts, Stime, Label
column_names = ['proto', 'dur', 'Spkts', 'Stime', 'Label']

def load_selected_columns(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(path, header=None, usecols=usecols, names=column_names, low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"Error al leer {filename}: {e}")
    return dfs

print("Cargando archivos...")
dfs = load_selected_columns(data_dir)
final_df = pd.concat(dfs, ignore_index=True)

# ----------------------------
# Preprocesamiento robusto
# ----------------------------
final_df['dur'] = pd.to_numeric(final_df['dur'], errors='coerce')
final_df['Spkts'] = pd.to_numeric(final_df['Spkts'], errors='coerce')
final_df = final_df.dropna(subset=['dur', 'Spkts', 'proto'])

# Definir plataformas nominales (top 5)
top_protocols = final_df['proto'].value_counts().nlargest(5).index
final_df['platform'] = final_df['proto'].apply(lambda p: p if p in top_protocols else "Others")

# Score ordinal basado en duración y tamaño
duration = final_df['dur'].clip(lower=0)
packets = final_df['Spkts'].clip(lower=1)
combined = np.log1p(duration) + np.log1p(packets)
final_df['alert'] = (combined > combined.median()).astype(int)

num_bins = min(5, combined.nunique())
if num_bins >= 3:
    final_df['score'] = pd.qcut(combined, q=num_bins, labels=range(1, num_bins + 1))
else:
    raise ValueError(f"No hay suficientes valores únicos para calcular score. Se encontraron solo {combined.nunique()}.")

# Confirmación binaria
final_df['confirmed'] = final_df['Label'].astype(int)

# Batches temporales
final_df['Stime'] = pd.to_datetime(final_df['Stime'], errors='coerce')
final_df = final_df.dropna(subset=['Stime']).sort_values('Stime').reset_index(drop=True)
final_df['batch'] = pd.qcut(final_df.index, 4, labels=["T1", "T2", "T3", "T4"])

fair_df = final_df[["platform", "alert", "score", "confirmed", "batch"]].copy()

# ----------------------------
# Métricas con bootstrap
# ----------------------------
def bootstrap_group_stat(df, group_col, stat_func, n_bootstrap=1000):
    means = {}
    stds = {}
    all_samples = {}
    for g in df[group_col].unique():
        vals = []
        subdf = df[df[group_col] == g]
        for _ in range(n_bootstrap):
            sample = subdf.sample(frac=1, replace=True)
            vals.append(stat_func(sample))
        means[g] = np.mean(vals)
        stds[g] = np.std(vals)
        all_samples[g] = vals  # guardar para test posterior
    return pd.Series(means), pd.Series(stds), pd.DataFrame(all_samples)

# φ_ind
def compute_phi_ind(df, group_col='platform'):
    p_group = df[group_col].value_counts(normalize=True).sort_index()
    p_group_alerts = df[df['alert'] == 1][group_col].value_counts(normalize=True).sort_index()
    return (p_group_alerts / p_group).fillna(0)

def bootstrap_phi_ind(df, group_col='platform', n_bootstrap=1000):
    values = []
    for _ in range(n_bootstrap):
        sample = df.sample(frac=1, replace=True)
        phi = compute_phi_ind(sample, group_col)
        values.append(phi)
    phi_df = pd.DataFrame(values)
    return phi_df.mean(), phi_df.std(), phi_df  # también se devuelve todo para análisis posterior

# Calcular φ_ind
mean_phi_ind, std_phi_ind, raw_phi_ind = bootstrap_phi_ind(fair_df)
phi_ind_df = pd.DataFrame({'mean': mean_phi_ind, 'std': std_phi_ind})
phi_ind_df.to_csv("phi_ind_bootstrap.csv")
raw_phi_ind.to_csv("phi_ind_raw.csv")  # para test posterior

mean_phi_ind.plot(kind='bar', yerr=std_phi_ind, capsize=4, title=r"$\phi_{\text{ind}}$ by protocol (95% CI)")
plt.xlabel("Protocol")
plt.ylabel("φ_ind")
plt.tight_layout()
plt.savefig("phi_ind_unsw.png")
plt.close()

# Calcular φ_sep
def f1_wrapper(df):
    return f1_score(df['confirmed'], df['alert'], zero_division=0)

mean_phi_sep, std_phi_sep, raw_phi_sep = bootstrap_group_stat(fair_df, 'platform', f1_wrapper)
phi_sep_df = pd.DataFrame({'mean': mean_phi_sep, 'std': std_phi_sep})
phi_sep_df.to_csv("phi_sep_bootstrap.csv")
raw_phi_sep.to_csv("phi_sep_raw.csv")  # para test posterior

mean_phi_sep.plot(kind='bar', yerr=std_phi_sep, capsize=4, title=r"$\phi_{\text{sep}}$ by protocol (95% CI)")
plt.xlabel("Protocol")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig("phi_sep_unsw.png")
plt.close()

# Calcular δ_cal
cal = fair_df.groupby(['score', 'platform'])['confirmed'].mean().unstack()
delta_cal = cal.max(axis=1) - cal.min(axis=1)
delta_cal.to_csv("delta_cal.csv")

delta_cal.plot(marker='o', title=r"$\delta_{\text{cal}}$ by score")
plt.ylabel("Δ P(confirmed=1 | score, platform)")
plt.xlabel("Score")
plt.tight_layout()
plt.savefig("delta_cal_unsw.png")
plt.close()