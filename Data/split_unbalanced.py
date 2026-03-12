import pandas as pd

df = pd.read_csv('Data/Initial data/heart_disease_data.csv',header=None)
size = len(df) // 4
print("Creating Non-IID (Unequal) Distributions for the 4 Hospitals...")

df = df.sort_values(by=18, ascending=True)
Hospital_A  = df.iloc[:size]

df = df.iloc[size:]

df = df.sort_values(by=[4,0], ascending=False)
Hospital_B = df.iloc[:size]

df = df.iloc[size:]

df = df.sort_values(by=[6,18], ascending=False)
Hospital_C = df.iloc[:size]

df = df.iloc[size:]

Hospital_D = df

Hospital_A = Hospital_A.sample(frac=1, random_state=42).reset_index(drop=True)
Hospital_B = Hospital_B.sample(frac=1, random_state=42).reset_index(drop=True)
Hospital_C = Hospital_C.sample(frac=1, random_state=42).reset_index(drop=True)
Hospital_D = Hospital_D.sample(frac=1, random_state=42).reset_index(drop=True)

Hospital_A.to_csv('Data/Unbalanced_split_data/Hospital_A.csv', index=False, header=False)
Hospital_B.to_csv('Data/Unbalanced_split_data/Hospital_B.csv', index=False, header=False) 
Hospital_C.to_csv('Data/Unbalanced_split_data/Hospital_C.csv', index=False, header=False)
Hospital_D.to_csv('Data/Unbalanced_split_data/Hospital_D.csv', index=False, header=False)

print("Data distribution completed and saved as CSV files for each hospital.")