import os
import pandas as pd
import numpy as np

# ----------------------------
# Configuración
# ----------------------------
data_dir = 'C:/Fairness/UNSW'
usecols = [4, 7, 17, 48]  # proto, dur, Spkts, Label
column_names = ['proto', 'dur', 'Spkts', 'Label']
ref_group = 'tcp'

# ----------------------------
# Carga de datos
# ----------------------------
dfs = []
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(path, header=None, usecols=usecols, names=column_names, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")

if not dfs:
    raise RuntimeError("No se pudieron cargar archivos del directorio.")

df = pd.concat(dfs, ignore_index=True)

# ----------------------------
# Preprocesamiento
# ----------------------------
df['proto'] = df['proto'].astype(str)
df['dur'] = pd.to_numeric(df['dur'], errors='coerce')
df['Spkts'] = pd.to_numeric(df['Spkts'], errors='coerce')
df['Label'] = pd.to_numeric(df['Label'], errors='coerce')

df = df.dropna(subset=['proto', 'dur', 'Spkts', 'Label'])

top_protocols = df['proto'].value_counts().nlargest(5).index
df['platform'] = df['proto'].apply(lambda p: p if p in top_protocols else "Others")

combined = (df['dur'].clip(lower=0).apply(lambda x: x + 1).apply(np.log) +
            df['Spkts'].clip(lower=1).apply(lambda x: x + 1).apply(np.log))
df['alert'] = (combined > combined.median()).astype(int)
df['confirmed'] = df['Label'].astype(int)

# ----------------------------
# Calcular SPD y EOD
# ----------------------------
alert_rate = df.groupby('platform')['alert'].mean()
spd = alert_rate - alert_rate.get(ref_group)

def recall(group):
    positives = group[group['confirmed'] == 1]
    if len(positives) == 0:
        return 0.0
    return positives['alert'].mean()

eod = df.groupby('platform').apply(recall) - recall(df[df['platform'] == ref_group])

# ----------------------------
# Combinar resultados y guardar CSV
# ----------------------------
comparison_df = pd.DataFrame({
    'SPD (vs tcp)': spd.round(4),
    'EOD (vs tcp)': eod.round(4)
}).sort_index()

comparison_df.to_csv("fairness_classic_metrics.csv")
print("CSV guardado como 'fairness_classic_metrics.csv'.")

# ----------------------------
# Generar tabla LaTeX
# ----------------------------
latex_lines = [
    "\\begin{table}[ht]",
    "\\centering",
    "\\caption{Classical fairness metrics (SPD and EOD) by platform, using TCP as reference.}",
    "\\label{tab:classic_fairness}",
    "\\begin{tabular}{lcc}",
    "\\hline",
    "\\textbf{Platform} & \\textbf{SPD (vs tcp)} & \\textbf{EOD (vs tcp)} \\\\",
    "\\hline"
]

for idx, row in comparison_df.iterrows():
    latex_lines.append(f"{idx} & {row['SPD (vs tcp)']:.4f} & {row['EOD (vs tcp)']:.4f} \\\\")

latex_lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

with open("fairness_classic_metrics.tex", "w") as f:
    f.write("\n".join(latex_lines))

print("Tabla LaTeX generada en 'fairness_classic_metrics.tex'")
