"""
Customer Segmentation Pipeline
--------------------------------
This script:
1. Loads raw retail data
2. Creates RFM features
3. Performs KMeans clustering
4. Evaluates clustering stability
5. Compares with Hierarchical clustering
6. Generates revenue analysis
7. Visualizes clusters using PCA
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.rfm import clean_data, create_rfm


# =====================================================
# Helper Functions
# =====================================================

def evaluate_cluster(rfm_df, label_column):
    """
    Prints summary statistics, cluster counts,
    and revenue distribution for given cluster labels.
    """
    print(f"\n=== {label_column} Summary ===")

    summary = rfm_df.groupby(label_column).mean()
    print("\nCluster Summary:")
    print(summary)

    print("\nCluster Counts:")
    print(rfm_df[label_column].value_counts())

    revenue = rfm_df.groupby(label_column)["Monetary"].sum()

    print("\nRevenue by Cluster:")
    print(revenue)

    print("\nRevenue Percentage:")
    print(revenue / revenue.sum() * 100)


# =====================================================
# Main Pipeline
# =====================================================

def main():

    # -------------------------------
    # 1️⃣ Load Data
    # -------------------------------
    df = pd.read_excel("data/raw/OnlineRetail.xlsx")

    # -------------------------------
    # 2️⃣ Data Cleaning + RFM Creation
    # -------------------------------
    df_clean = clean_data(df)
    rfm = create_rfm(df_clean)

    # Remove CustomerID for clustering
    rfm_features = rfm.drop(columns=["CustomerID"])

    # -------------------------------
    # 3️⃣ Feature Scaling
    # -------------------------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # -------------------------------
    # 4️⃣ KMeans Clustering
    # -------------------------------
    print("\n--- KMeans Clustering ---")

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    kmeans_sil = silhouette_score(rfm_scaled, rfm["Cluster"])
    print("KMeans Silhouette Score:", round(kmeans_sil, 3))

    evaluate_cluster(rfm, "Cluster")

    # -------------------------------
    # 5️⃣ Stability Test
    # -------------------------------
    print("\n--- Stability Test (Different Random Seeds) ---")

    for seed in [0, 10, 20, 42, 100]:
        model = KMeans(n_clusters=3, random_state=seed)
        labels = model.fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, labels)
        print(f"Seed {seed} → Silhouette: {round(score, 3)}")

    # -------------------------------
    # 6️⃣ Hierarchical Clustering
    # -------------------------------
    print("\n--- Hierarchical Clustering ---")

    hierarchical = AgglomerativeClustering(n_clusters=3)
    rfm["Hier_Cluster"] = hierarchical.fit_predict(rfm_scaled)

    hier_sil = silhouette_score(rfm_scaled, rfm["Hier_Cluster"])
    print("Hierarchical Silhouette Score:", round(hier_sil, 3))

    evaluate_cluster(rfm, "Hier_Cluster")

    # -------------------------------
    # 7️⃣ PCA Visualization
    # -------------------------------
    print("\n--- PCA Visualization ---")

    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)

    plt.figure()
    plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm["Cluster"])
    plt.title("Customer Segments (KMeans)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # -------------------------------
    # 8️⃣ Save Processed Output
    # -------------------------------
    rfm.to_csv("data/processed/rfm_with_clusters.csv", index=False)
    print("\nProcessed dataset saved to data/processed/")

    # -------------------------------
    # 9️⃣ Revenue Percentile Segmentation
    # -------------------------------
    print("\n--- Revenue Percentile Segmentation ---")

    # Calculate percentile rank based on Monetary value
    rfm["RevenuePercentile"] = rfm["Monetary"].rank(pct=True)

    # Default tier
    rfm["Tier"] = "Regular"

    # Top 1% = Ultra VIP
    rfm.loc[rfm["RevenuePercentile"] > 0.99, "Tier"] = "Ultra VIP"

    # Top 5% (excluding top 1%) = Premium
    rfm.loc[
        (rfm["RevenuePercentile"] > 0.95) &
        (rfm["RevenuePercentile"] <= 0.99),
        "Tier"
    ] = "Premium"

    # Tier distribution
    print("\nTier Counts:")
    print(rfm["Tier"].value_counts())

    # Revenue distribution per tier
    tier_revenue = rfm.groupby("Tier")["Monetary"].sum()

    print("\nRevenue by Tier:")
    print(tier_revenue)

    print("\nRevenue Percentage by Tier:")
    print(tier_revenue / tier_revenue.sum() * 100)

    # -------------------------------
    # 🔟 DBSCAN (Outlier Detection)
    # -------------------------------
    from sklearn.cluster import DBSCAN

    print("\n--- DBSCAN Clustering (Outlier Detection) ---")

    # Tune eps carefully (try 1.0–1.5 depending on scaling)
    dbscan = DBSCAN(eps=1.2, min_samples=5)
    rfm["DBSCAN"] = dbscan.fit_predict(rfm_scaled)

    # Count clusters and noise
    print("\nDBSCAN Label Counts:")
    print(rfm["DBSCAN"].value_counts())

    # Number of outliers
    n_outliers = (rfm["DBSCAN"] == -1).sum()
    print(f"\nNumber of detected outliers: {n_outliers}")

    # Revenue of outliers
    if n_outliers > 0:
        outlier_revenue = rfm[rfm["DBSCAN"] == -1]["Monetary"].sum()
        print("\nOutlier Revenue:")
        print(outlier_revenue)
        print("\nOutlier Revenue Percentage:")
        print(outlier_revenue / rfm["Monetary"].sum() * 100)

    plt.figure()
    plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm["DBSCAN"])
    plt.title("DBSCAN Outlier Detection (PCA View)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


# =====================================================
# Entry Point
# =====================================================


if __name__ == "__main__":
    main()
