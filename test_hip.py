import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def rank_biserial(U, n1, n2):
    """Compute rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * U) / (n1 * n2)

def interpret_effect(r):
    """Interpret effect size magnitude (Cohen-like thresholds for r)."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"

results = []

# ----------------------------
# 1. Detection Separation
# ----------------------------
print("== φ_sep: Detection quality difference ==")
phi_sep = pd.read_csv("phi_sep_raw.csv")

group1 = phi_sep['ospf'].dropna()
group2 = phi_sep['tcp'].dropna()

print("Samples for φ_sep:")
print("ospf:", group1.describe())
print("tcp:", group2.describe())
print("------")

if len(group1) > 0 and len(group2) > 0:
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    r = rank_biserial(stat, len(group1), len(group2))
    interp = interpret_effect(r)
    print(f"OSPF vs TCP: U = {stat:.2f}, p = {p:.2e}, r = {r:.4f} ({interp})")
    results.append({
        'metric': 'phi_sep', 'comparison': 'OSPF vs TCP',
        'U': stat, 'p_value': p, 'effect_size_r': r, 'interpretation': interp,
        'n1': len(group1), 'n2': len(group2)
    })
else:
    print("Insufficient data to compare OSPF vs TCP for φ_sep.")

# ----------------------------
# 2. Operational Independence
# ----------------------------
print("\n== φ_ind: Alert distribution difference ==")
phi_ind = pd.read_csv("phi_ind_raw.csv")

group1 = phi_ind['tcp'].dropna()
group2 = phi_ind['ospf'].dropna()

print("Samples for φ_ind:")
print("tcp:", group1.describe())
print("ospf:", group2.describe())
print("------")

if len(group1) > 0 and len(group2) > 0:
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    r = rank_biserial(stat, len(group1), len(group2))
    interp = interpret_effect(r)
    print(f"TCP vs OSPF: U = {stat:.2f}, p = {p:.2e}, r = {r:.4f} ({interp})")
    results.append({
        'metric': 'phi_ind', 'comparison': 'TCP vs OSPF',
        'U': stat, 'p_value': p, 'effect_size_r': r, 'interpretation': interp,
        'n1': len(group1), 'n2': len(group2)
    })
else:
    print("Insufficient data to compare TCP vs OSPF for φ_ind.")

# ----------------------------
# 3. δ_cal: Calibration difference
# ----------------------------
print("\n== δ_cal: Score consistency across platforms ==")
delta_cal = pd.read_csv("delta_cal.csv", index_col=0).squeeze()

delta1 = delta_cal.loc[1]
delta5 = delta_cal.loc[5]

print(f"δ_cal score 1: {delta1:.4f}")
print(f"δ_cal score 5: {delta5:.4f}")
print(f"Absolute difference |δ_cal(1) - δ_cal(5)|: {abs(delta1 - delta5):.4f}")

# Note: δ_cal values are scalar point estimates (not bootstrap distributions).
# A Mann-Whitney U test on replicated scalars is not meaningful.
# Instead, we report the magnitude of the difference directly.
# For completeness, we also perform the test on replicated samples,
# acknowledging that the resulting p-value is trivially significant
# and should be interpreted solely as confirmation of non-equality.
delta1_dist = np.array([delta1] * 1000)
delta5_dist = np.array([delta5] * 1000)

if delta1 != delta5:
    stat, p = mannwhitneyu(delta1_dist, delta5_dist, alternative='two-sided')
    print(f"Score 1 vs 5: U = {stat:.2f}, p = {p:.2e}")
    print("Note: This test is performed on replicated scalar values (variance = 0).")
    print("The p-value is trivially significant and should be interpreted as")
    print("confirmation of non-equality, not as evidence of distributional difference.")
    print(f"The substantive evidence comes from the magnitude: |Δ| = {abs(delta1 - delta5):.4f}")
    results.append({
        'metric': 'delta_cal', 'comparison': 'Score 1 vs Score 5',
        'U': stat, 'p_value': p, 'effect_size_r': np.nan,
        'interpretation': f'|Delta| = {abs(delta1 - delta5):.4f} (direct magnitude)',
        'n1': 1000, 'n2': 1000
    })
else:
    print("δ_cal(1) == δ_cal(5): no difference to test.")

# ----------------------------
# Summary table
# ----------------------------
print("\n" + "=" * 70)
print("SUMMARY OF STATISTICAL TESTS")
print("=" * 70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("hypothesis_test_results.csv", index=False)
print("\nResults saved to hypothesis_test_results.csv")
