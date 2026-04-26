import pandas as pd
df1 = pd.read_csv("D:/PhD_Deep_Learning/Local Government/Train_Data/train_data_updated_750.csv")
print(len(df1['Location'].unique()))
df1['Date'] = pd.to_datetime(df1['Date'])

# Perform boolean indexing to select data up to the specific date
df = df1[df1['Date'] <= '2023-12-01']
print(len(df))
selected_feat = ['T2M', 'T2MWET', 'TS',
       'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M', 'PRECTOTCORR', 'PS',
       'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX',
       'WS50M_MIN', 'WS50M_RANGE']
# selected_feat = ['T2M', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN', 'RH2M',
#         'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS50M', 'WS50M_MAX',
#         'WS50M_MIN']
data1 = df[selected_feat]
print(data1.columns)
# mean_data = data.mean()

mean_values = data1.mean()
std_values = data1.std()
print(mean_values)
print(std_values)
# Save mean and standard deviation to CSV files

pd.DataFrame(mean_values).T.to_csv('./Train_Data/mean_values.csv', index=False)
pd.DataFrame(std_values).T.to_csv('./Train_Data/std_values.csv', index=False)

# Normalize the data
normalized_data1 = (data1 - mean_values) / std_values

df[selected_feat] = normalized_data1[selected_feat]

df.to_csv('./Train_Data/normalized_data_750.csv', index=False)
