import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from load_data import load_data
from k_means import k_means

def test_k_means(X, k_values):
    """
    Test K-means algorithm with different k values.
    """
    for k in k_values:
        print(f"\nTesting K-means with k={k}")
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'K-means Clustering (k={k})')
        
        # Run K-means
        clusters, centroids = k_means(X, k)
        
        # Plot results
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.title('Original Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # Plot standardized results
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clusters_scaled, centroids_scaled = k_means(X_scaled, k)
        
        plt.subplot(1, 2, 2)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_scaled, cmap='viridis')
        plt.scatter(centroids_scaled[:, 0], centroids_scaled[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.title('Standardized Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # Read the data
    filename = "assignments\PS3-data\q4\X.dat" 
    X = load_data(filename)
    
    # Test K-means with k=3 and k=4
    test_k_means(X, [3, 4])