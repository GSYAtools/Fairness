import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Cargar el dataset sesgado
df = pd.read_csv("synthetic_fairness_data_biased.csv")

sns.set(style="whitegrid")

# ----------------------------
# 1. Temporal F1-score por platform
# ----------------------------

f1_records = []

for (batch, platform), group in df.groupby(['batch', 'platform']):
    f1 = f1_score(group['confirmed'], group['alert'], zero_division=0)
    f1_records.append({'Batch': batch, 'Platform': platform, 'F1': f1})

f1_df = pd.DataFrame(f1_records)
f1_df.to_csv("phi_sep_temporal_platform.csv", index=False)

# Graficar evolución temporal
plt.figure(figsize=(8, 5))
sns.lineplot(data=f1_df, x='Batch', y='F1', hue='Platform', marker='o')
plt.title("Temporal Evolution of Detection Separation (\u03C6_sep) by Platform")
plt.ylabel("F1-score")
plt.xlabel("Time Batch")
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("phi_sep_temporal_platform.png", dpi=300)
plt.close()

# ----------------------------
# 2. Temporal δ_cal por score (dispersión entre plataformas)
# ----------------------------

delta_records = []

for (batch, score), group in df.groupby(['batch', 'score']):
    pivot = group.groupby('platform')['confirmed'].mean()
    if len(pivot) > 1:
        delta = pivot.max() - pivot.min()
    else:
        delta = 0.0
    delta_records.append({'Batch': batch, 'Score': score, 'Delta_Cal': delta})

delta_df = pd.DataFrame(delta_records)
delta_df.to_csv("delta_cal_temporal_score.csv", index=False)

# Graficar evolución temporal
plt.figure(figsize=(8, 5))
sns.lineplot(data=delta_df, x='Batch', y='Delta_Cal', hue='Score', marker='o')
plt.title("Temporal Evolution of Calibration Sufficiency (\u03B4_cal)")
plt.ylabel("Δ P(confirmed=1 | score, platform)")
plt.xlabel("Time Batch")
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("delta_cal_temporal_score.png", dpi=300)
plt.close()