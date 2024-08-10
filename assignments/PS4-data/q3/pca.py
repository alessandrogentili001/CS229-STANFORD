import numpy as np 

def pca(X):
    """
    Perform Principal Component Analysis (PCA) using SVD.
    
    Args:
    X (numpy.ndarray): Input data matrix of shape (n_features, n_samples)
    
    Returns:
    numpy.ndarray: Principal components (eigenvectors) of shape (n_features, n_features)
    """
    U, _, _ = np.linalg.svd(X @ X.T)
    return U
