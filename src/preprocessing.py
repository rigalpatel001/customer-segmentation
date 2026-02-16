
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    
    # Drop CustomerID if exists
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    
    # Encode Gender if present
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    return scaled_data, df.columns
