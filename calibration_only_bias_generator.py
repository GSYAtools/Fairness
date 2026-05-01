import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(88)

# Total number of samples
n_samples = 40000

# Define categories
platforms = ['Windows', 'Linux', 'IoT']
scores = [1, 2, 3, 4, 5]

# Generate platform with biased visibility (same as full-bias scenario)
data = pd.DataFrame({
    'platform': np.random.choice(platforms, size=n_samples, p=[0.7, 0.2, 0.1]),
})

# Alert generation: NEUTRAL (uniform across platforms)
data['alert'] = (np.random.rand(n_samples) < 0.4).astype(int)

# Proxy ground truth: BIASED (platform-dependent, same as full-bias)
# This creates the calibration gap: P(confirmed | score, platform) differs
def generate_confirmed(row):
    return int(np.random.rand() < {
        'Windows': 0.7,
        'Linux': 0.65,
        'IoT': 0.5
    }[row['platform']])
data['confirmed'] = data.apply(generate_confirmed, axis=1)

# Score assignment: BIASED (platform-dependent semantic drift)
# Same drift pattern as full-bias scenario
def generate_score(row):
    base = np.random.choice(scores, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    if row['platform'] == 'Windows':
        return base  # calibrated
    elif row['platform'] == 'Linux':
        return max(1, base - np.random.choice([0, 1], p=[0.7, 0.3]))  # conservative
    else:  # IoT
        return min(5, base + np.random.choice([0, 1], p=[0.3, 0.7]))  # inflated
data['score'] = data.apply(generate_score, axis=1)

# Temporal batches
data['batch'] = pd.qcut(data.index, q=4, labels=["T1", "T2", "T3", "T4"])

# Save
data.to_csv("synthetic_fairness_data_calibration_only.csv", index=False)
print("Generated: synthetic_fairness_data_calibration_only.csv")
print(f"N = {len(data)}")
print("\nAlert rate by platform (should be ~0.4 for all):")
print(data.groupby('platform')['alert'].mean())
print("\nConfirmed rate by platform (should differ):")
print(data.groupby('platform')['confirmed'].mean())
print("\nScore distribution by platform:")
print(data.groupby('platform')['score'].value_counts().unstack().fillna(0))
