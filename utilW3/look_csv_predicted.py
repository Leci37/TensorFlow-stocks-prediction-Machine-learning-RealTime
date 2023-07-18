
import pandas as pd


bars = pd.read_csv('outputs/model_info/U_5min_loss0.csv', index_col=0,   sep='\t')

df_details = pd.DataFrame()
df_details['count'] = bars.value_counts().to_frame()
df_details['per%'] = bars.value_counts(normalize=True).mul(100).round(2)

print('End')