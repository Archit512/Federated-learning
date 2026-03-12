import pandas as pd

df = pd.read_excel("Data/Initial data/heart_disease_data.xlsx")
label = df.iloc[:, 0]
features = df.iloc[:, 1:]

df = pd.concat([features, label], axis=1)
df.to_csv("Data/Initial data/heart_disease_data.csv", index=False, header=False)
