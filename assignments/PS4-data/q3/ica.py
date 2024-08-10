import numpy as np 

def ica(X):
    """
    Perform Independent Component Analysis (ICA) using gradient descent.
    
    Args:
    X (numpy.ndarray): Input data matrix of shape (n_features, n_samples)
    
    Returns:
    numpy.ndarray: Unmixing matrix W of shape (n_features, n_features)
    """
    n, m = X.shape
    chunk = 100 # batch size
    alpha = 0.0005 # step size 
    W = np.eye(n) # initialize W
    
    for iter in range(10):
        print(f"Iteration {iter + 1}/10")
        X = X[:, np.random.permutation(m)] # shuffle data
        
        for i in range(m // chunk):
            Xc = X[:, i*chunk:(i+1)*chunk] # select a batch
            dW = (1 - 2 / (1 + np.exp(-W @ Xc))) @ Xc.T + chunk * np.linalg.inv(W).T # compute gradient
            W = W + alpha * dW # update W (maximization)
    
    return W
