import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# ----------------------------
# Configuración gráfica
# ----------------------------
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

# ----------------------------
# Ruta de datos
# ----------------------------
traffic_labeling_dir = 'C:/Fairness/TrafficLabeling'
required_columns = ["Flow ID", "Source IP", "Source Port", "Destination IP",
                    "Destination Port", "Protocol", "Timestamp", "Flow Duration",
                    "Total Fwd Packets", "Label"]

# ----------------------------
# Cargar CSVs
# ----------------------------
def load_csv_files(directory, required_columns):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(path, encoding='latin1', low_memory=False)
                df.columns = df.columns.str.strip()
                if all(col in df.columns for col in required_columns):
                    dfs.append(df[required_columns].copy())
            except Exception as e:
                print(f"Error al leer {filename}: {e}")
    return dfs

print("Cargando archivos...")
dfs = load_csv_files(traffic_labeling_dir, required_columns)
final_df = pd.concat(dfs, ignore_index=True)

# ----------------------------
# Procesamiento de columnas
# ----------------------------

# Convertir Protocol a numérico limpio
final_df['Protocol'] = pd.to_numeric(final_df['Protocol'], errors='coerce').astype('Int64')
final_df = final_df.dropna(subset=['Protocol'])

# Identificar los 5 protocolos más frecuentes
top_protocols = final_df['Protocol'].value_counts().nlargest(5).index

# Asignar nombre "Protocol_xxx" o "Others"
def map_or_other(proto):
    if proto in top_protocols:
        return f"Protocol_{proto}"
    else:
        return 'Others'

final_df['platform'] = final_df['Protocol'].apply(map_or_other)

# ----------------------------
# Generar campos artificiales para fairness
# ----------------------------

duration = final_df['Flow Duration'].clip(lower=0)
packets = final_df['Total Fwd Packets'].clip(lower=1)
combined = np.log1p(duration) + np.log1p(packets)

# Crear alerta binaria
final_df['alert'] = (combined > combined.median()).astype(int)

# Crear score basado en cuantiles
final_df['score'] = pd.qcut(combined, 5, labels=[1, 2, 3, 4, 5])

# Confirmed basado en etiquetas
final_df['confirmed'] = final_df['Label'].apply(lambda x: 0 if 'benign' in str(x).lower() else 1)

# Procesar Timestamp y generar batches
final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], errors='coerce')
final_df = final_df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
final_df['batch'] = pd.qcut(final_df.index, 4, labels=["T1", "T2", "T3", "T4"])

# Subset útil para métricas
fair_df = final_df[["platform", "alert", "score", "confirmed", "batch"]].copy()
fair_df.to_csv("cic_fairness_ready.csv", index=False)

# ----------------------------
# Cálculo de métricas
# ----------------------------

# φ_ind: Operational Independence
def compute_phi_ind(df, group_col='platform'):
    p_group = df[group_col].value_counts(normalize=True).sort_index()
    p_group_alerts = df[df['alert'] == 1][group_col].value_counts(normalize=True).sort_index()
    return (p_group_alerts / p_group).fillna(0)

phi_ind = compute_phi_ind(fair_df)
phi_ind.to_csv("phi_ind_cic.csv")
phi_ind.plot(kind='bar', title=r"$\phi_{\text{ind}}$ by platform")
plt.ylabel("φ_ind")
plt.tight_layout()
plt.savefig("phi_ind_realdata.png")
plt.close()

# φ_sep: Detection Separation
phi_sep = fair_df.groupby("platform").apply(lambda g: f1_score(g['confirmed'], g['alert'], zero_division=0))
phi_sep.to_csv("phi_sep_cic.csv")
phi_sep.plot(kind='bar', title=r"$\phi_{\text{sep}}$ by platform")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig("phi_sep_realdata.png")
plt.close()

# δ_cal: Calibration Sufficiency
cal = fair_df.groupby(['score', 'platform'])['confirmed'].mean().unstack()
delta_cal = cal.max(axis=1) - cal.min(axis=1)
delta_cal.to_csv("delta_cal_cic.csv")

delta_cal.plot(marker='o', title=r"$\delta_{\text{cal}}$ by score")
plt.ylabel("Δ P(confirmed=1 | score, platform)")
plt.xlabel("Score")
plt.tight_layout()
plt.savefig("delta_cal_realdata.png")
plt.close()

