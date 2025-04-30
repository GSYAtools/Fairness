import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Total number of samples
n_samples = 40000

# Define categories
platforms = ['Windows', 'Linux', 'IoT']
scores = [1, 2, 3, 4, 5]

# Generate platform with biased visibility
data = pd.DataFrame({
    'platform': np.random.choice(platforms, size=n_samples, p=[0.7, 0.2, 0.1]),
})

# Generate confirmed label (proxy ground truth) with variation by platform
def generate_confirmed(row):
    return int(np.random.rand() < {
        'Windows': 0.7,
        'Linux': 0.65,
        'IoT': 0.5  # Menor tasa de eventos reales en IoT
    }[row['platform']])
data['confirmed'] = data.apply(generate_confirmed, axis=1)

# Generate alerts with overalerting on IoT (independent of confirmation)
def generate_alert(row):
    return int(np.random.rand() < {
        'Windows': 0.3,
        'Linux': 0.4,
        'IoT': 0.9  # Alerting bias: IoT triggers many more alerts
    }[row['platform']])
data['alert'] = data.apply(generate_alert, axis=1)

# Generate score with semantic drift by platform
def generate_score(row):
    base = np.random.choice(scores, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    if row['platform'] == 'Windows':
        return base  # calibrado
    elif row['platform'] == 'Linux':
        return max(1, base - np.random.choice([0, 1], p=[0.7, 0.3]))  # más conservador
    else:  # IoT
        return min(5, base + np.random.choice([0, 1], p=[0.3, 0.7]))  # inflado
data['score'] = data.apply(generate_score, axis=1)

# Añadir batches temporales
data['batch'] = pd.qcut(data.index, q=4, labels=["T1", "T2", "T3", "T4"])

# Guardar a CSV
data.to_csv("synthetic_fairness_data_biased.csv", index=False)