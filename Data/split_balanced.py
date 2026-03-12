import pandas as pd

df = pd.read_csv('Data/Initial data/heart_disease_data.csv')
size = len(df) // 4

print("Creating Balanced/Randomised Distributions for the 4 Hospitals...")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

Hospital_A  = df.iloc[:size]
Hospital_B = df.iloc[size:2*size]
Hospital_C = df.iloc[2*size:3*size]
Hospital_D = df.iloc[3*size:4*size]

Hospital_A.to_csv('Data/Balanced_split_data/Hospital_A.csv', index=False, header=False)
Hospital_B.to_csv('Data/Balanced_split_data/Hospital_B.csv', index=False, header=False)
Hospital_C.to_csv('Data/Balanced_split_data/Hospital_C.csv', index=False, header=False)
Hospital_D.to_csv('Data/Balanced_split_data/Hospital_D.csv', index=False, header=False)

print("Data distribution completed and saved as CSV files for each hospital.")