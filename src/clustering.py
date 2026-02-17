from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def run_kmeans(data, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, score


def run_hierarchical(data, k=3):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, score
