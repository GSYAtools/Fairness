import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(123)

# Total number of samples
n_samples = 40000

# Define categories
platforms = ['Windows', 'Linux', 'IoT']
scores = [1, 2, 3, 4, 5]

# Generate platform with uniform visibility (true neutral)
data_neutral = pd.DataFrame({
    'platform': np.random.choice(platforms, size=n_samples, p=[1/3, 1/3, 1/3]),
})

# Generate unbiased alert decision (same probability for all)
data_neutral['alert'] = (np.random.rand(n_samples) < 0.4).astype(int)

# Generate scores with fixed, symmetric distribution
data_neutral['score'] = np.random.choice(scores, size=n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])

# Uniform proxy ground truth (same for all platforms)
data_neutral['confirmed'] = (np.random.rand(n_samples) < 0.6).astype(int)

# Add artificial time batches (to test drift)
data_neutral['batch'] = pd.qcut(data_neutral.index, q=4, labels=["T1", "T2", "T3", "T4"])

# Save to CSV
data_neutral.to_csv("synthetic_fairness_data_neutral.csv", index=False)