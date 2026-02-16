
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("Cluster Visualization (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
