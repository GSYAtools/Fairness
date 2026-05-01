import os
import pandas as pd
import numpy as np

data_dir = 'D:/Fairness/UNSW'
usecols = [4, 7, 17, 29, 48]
column_names = ['proto', 'dur', 'Spkts', 'Stime', 'Label']

dfs = []
for f in os.listdir(data_dir):
    if f.endswith('.csv'):
        dfs.append(pd.read_csv(os.path.join(data_dir, f), header=None, usecols=usecols, names=column_names, low_memory=False))

df = pd.concat(dfs, ignore_index=True)
print('N bruto:', len(df))

df['dur'] = pd.to_numeric(df['dur'], errors='coerce')
df['Spkts'] = pd.to_numeric(df['Spkts'], errors='coerce')
df = df.dropna(subset=['dur', 'Spkts', 'proto'])

df['Stime'] = pd.to_datetime(df['Stime'], errors='coerce')
df = df.dropna(subset=['Stime'])
print('N tras filtrado:', len(df))