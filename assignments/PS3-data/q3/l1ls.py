import numpy as np

def l1ls(X, y, lambda_val):
    
    # Get dimensions of X
    m, n = X.shape
    
    # Initialize theta as a zero vector
    theta = np.zeros(n)
    
    # Initialize old_theta as a vector of ones (for first iteration)
    old_theta = np.ones(n)
    
    # Iterate until convergence
    while np.linalg.norm(theta - old_theta) > 1e-5:
        old_theta = theta
        
        # Iterate through each feature
        for i in range(n):
            
            # Temporarily set theta[i] to 0 for calculations
            theta[i] = 0
            
            # Compute two possible values for theta[i]
            theta_i = np.zeros(2)
            theta_i[0] = max((-X[:, i].dot(X.dot(theta) - y) - lambda_val) / (X[:, i].dot(X[:, i])), 0)
            theta_i[1] = min((-X[:, i].dot(X.dot(theta) - y) + lambda_val) / (X[:, i].dot(X[:, i])), 0)
            
            # Compute objective values for both possible theta[i]
            obj_theta = np.zeros(2)
            for j in range(2):
                theta[i] = theta_i[j]
                obj_theta[j] = 0.5 * np.linalg.norm(X.dot(theta) - y)**2 + lambda_val * np.linalg.norm(theta, 1)
            
            # Choose theta[i] that minimizes the objective
            min_ind = np.argmin(obj_theta)
            theta[i] = theta_i[min_ind]
    
    return theta