import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove negative or zero values
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    return df


def create_rfm(df: pd.DataFrame) -> pd.DataFrame:

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Create TotalPrice
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Define snapshot date (one day after last purchase)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Aggregate per customer
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    return rfm.reset_index()
