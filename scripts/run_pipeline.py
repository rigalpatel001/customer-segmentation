
from src.preprocessing import load_dataset, preprocess_data
from src.clustering import run_kmeans, run_hierarchical, run_dbscan
from src.evaluate import evaluate_clustering
from src.visualize import plot_clusters

def main():
    df = load_dataset("data/raw/Mall_Customers.csv")
    X_scaled, feature_names = preprocess_data(df)
    
    print("\n--- Running KMeans ---")
    model, labels = run_kmeans(X_scaled, n_clusters=3)
    metrics = evaluate_clustering(X_scaled, labels)
    print(metrics)
    plot_clusters(X_scaled, labels)
    
    print("\n--- Running Hierarchical Clustering ---")
    model, labels = run_hierarchical(X_scaled, n_clusters=3)
    metrics = evaluate_clustering(X_scaled, labels)
    print(metrics)
    plot_clusters(X_scaled, labels)
    
    print("\n--- Running DBSCAN ---")
    model, labels = run_dbscan(X_scaled)
    metrics = evaluate_clustering(X_scaled, labels)
    print(metrics)
    plot_clusters(X_scaled, labels)

if __name__ == "__main__":
    main()
