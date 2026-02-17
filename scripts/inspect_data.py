import pandas as pd

file_path = "data/raw/OnlineRetail.xlsx"  # replace with your actual filename

df = pd.read_excel(file_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
