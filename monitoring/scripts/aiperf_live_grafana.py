import pandas as pd
df = pd.read_csv('/media/works/BurstGPT_without_fails_3.csv')
df_sorted = df.sort_values('Timestamp')
df_sorted['bucket_15min'] = (df_sorted['Timestamp'] // 900).astype(int)

steady = df_sorted[df_sorted['bucket_15min'] == 21661]
burst = df_sorted[df_sorted['bucket_15min'] == 23832]

print("Steady avg response tokens:", steady['Response tokens'].mean())
print("Burst avg response tokens:", burst['Response tokens'].mean())