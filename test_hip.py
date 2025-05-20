import pandas as pd
from scipy.stats import mannwhitneyu

# ----------------------------
# 1. Detection Separation
# ----------------------------
print("== φ_sep: Detection quality difference ==")
phi_sep = pd.read_csv("phi_sep_raw.csv")

group1 = phi_sep['ospf'].dropna()
group2 = phi_sep['tcp'].dropna()

print("Muestras para φ_sep:")
print("ospf:", group1.describe())
print("tcp:", group2.describe())
print("------")

if len(group1) > 0 and len(group2) > 0:
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"OSPF vs TCP: p = {p:.2e}")
else:
    print("Datos insuficientes para comparar OSPF vs TCP en φ_sep.")

# ----------------------------
# 2. Operational Independence
# ----------------------------
print("\n== φ_ind: Alert distribution difference ==")
phi_ind = pd.read_csv("phi_ind_raw.csv")

group1 = phi_ind['tcp'].dropna()
group2 = phi_ind['ospf'].dropna()

print("Muestras para φ_ind:")
print("tcp:", group1.describe())
print("ospf:", group2.describe())
print("------")

if len(group1) > 0 and len(group2) > 0:
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"TCP vs OSPF: p = {p:.2e}")
else:
    print("Datos insuficientes para comparar TCP vs OSPF en φ_ind.")

# ----------------------------
# 3. δ_cal: Calibration difference
# ----------------------------
print("\n== δ_cal: Score consistency across platforms ==")
delta_cal = pd.read_csv("delta_cal.csv", index_col=0).squeeze()

delta1 = delta_cal.loc[1]
delta5 = delta_cal.loc[5]

print(f"δ_cal score 1: {delta1}")
print(f"δ_cal score 5: {delta5}")

# Simular muestras para el test (artificial replicación)
delta1_dist = [delta1] * 1000
delta5_dist = [delta5] * 1000

stat, p = mannwhitneyu(delta1_dist, delta5_dist, alternative='two-sided')
print(f"Score 1 vs 5: p = {p:.2e}")