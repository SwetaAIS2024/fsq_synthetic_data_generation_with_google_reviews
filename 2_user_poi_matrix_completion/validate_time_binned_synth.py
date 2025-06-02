import pandas as pd
import matplotlib.pyplot as plt

# Load real and synthetic check-ins
df_real = pd.read_csv('../../fsq_original_dataset/singapore_checkins.txt', sep='\t', names=['user_id', 'poi_id', 'timestamp'])
df_synth = pd.read_csv('synthetic_checkins_time_binned.csv')

def extract_month(ts):
    try:
        return pd.to_datetime(ts).strftime('%Y-%m')
    except:
        return 'unknown'

df_real['month'] = df_real['timestamp'].apply(extract_month)
df_synth['month'] = df_synth['time_bin']

# Compare monthly distribution
real_monthly = df_real['month'].value_counts().sort_index()
synth_monthly = df_synth['month'].value_counts().sort_index()

plt.figure(figsize=(12,6))
plt.plot(real_monthly.index, real_monthly.values, label='Real', marker='o')
plt.plot(synth_monthly.index, synth_monthly.values, label='Synthetic', marker='x')
plt.title('Monthly Check-in Distribution: Real vs Synthetic')
plt.xlabel('Month')
plt.ylabel('Number of Check-ins')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('time_binned_monthly_distribution.png')
plt.show()

print('Validation complete. Plot saved as time_binned_monthly_distribution.png')
