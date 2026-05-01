import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(77)

# Total number of samples
n_samples = 40000

# Define categories
platforms = ['Windows', 'Linux', 'IoT']
scores = [1, 2, 3, 4, 5]

# Generate platform with biased visibility (same as full-bias scenario)
data = pd.DataFrame({
    'platform': np.random.choice(platforms, size=n_samples, p=[0.7, 0.2, 0.1]),
})

# Alert generation: BIASED (same as full-bias scenario)
# IoT is heavily overalerted relative to its prevalence
def generate_alert(row):
    return int(np.random.rand() < {
        'Windows': 0.3,
        'Linux': 0.4,
        'IoT': 0.9
    }[row['platform']])
data['alert'] = data.apply(generate_alert, axis=1)

# Proxy ground truth: NEUTRAL (uniform across platforms)
data['confirmed'] = (np.random.rand(n_samples) < 0.6).astype(int)

# Score assignment: NEUTRAL (no platform-dependent drift)
data['score'] = np.random.choice(scores, size=n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])

# Temporal batches
data['batch'] = pd.qcut(data.index, q=4, labels=["T1", "T2", "T3", "T4"])

# Save
data.to_csv("synthetic_fairness_data_alert_only.csv", index=False)
print("Generated: synthetic_fairness_data_alert_only.csv")
print(f"N = {len(data)}, Alert rate by platform:")
print(data.groupby('platform')['alert'].mean())
