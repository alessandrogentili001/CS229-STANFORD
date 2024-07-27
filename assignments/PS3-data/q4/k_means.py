import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data

def k_means(X, k):
    m, n = X.shape
    old_centroids = np.zeros((k, n))
    centroids = X[np.random.randint(m, size=k), :] # select randomly k centroids from X
    
    clusters = np.zeros(m, dtype=int)
    
    # repeat untill convergence 
    while np.linalg.norm(old_centroids - centroids) > 1e-15:
        old_centroids = centroids
        
        # Compute cluster assignments
        for i in range(m):
            dists = np.sum((X[i] - centroids)**2, axis=1)
            clusters[i] = np.argmin(dists)
        
        # Plot results dinamically 
        # plt.clf()
        # plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.draw()
        # plt.pause(0.5)
        
        # Compute new cluster centroids
        for i in range(k):
            centroids[i] = np.mean(X[clusters == i], axis=0)
    
    return clusters, centroids

if __name__ == "__main__":
    # Read the data
    filename = r"assignments\PS3-data\q4\X.dat" 
    X = load_data(filename)
    
    # Test K-means with k=3 and k=4
    k = 3
    k_means(X, k)
    k = 4
    k_means(X, k)