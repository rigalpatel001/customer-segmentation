
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X, labels):
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    return {
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score
    }
