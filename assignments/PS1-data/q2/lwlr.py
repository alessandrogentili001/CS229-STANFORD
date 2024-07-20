import numpy as np

def lwlr(X_train, y_train, x, tau):
    
    # read the data shape 
    m, n = X_train.shape
    
    # initialize parameters 
    theta = np.zeros(n)
    
    # Compute weights
    w = np.exp(- np.sum((X_train - x) ** 2, axis=1) / (2 * tau))
    
    # initilaize the gradient and perform Newton's method until the norm of the gradient is small enough 
    g = np.ones(n)
    while np.linalg.norm(g) > 1e-6:
        
        # compute the hypotesis function 
        h = 1 / (1 + np.exp(-X_train.dot(theta)))
        
        # compute the weighted gradient
        g = X_train.T.dot(w * (y_train - h)) - 1e-4 * theta
        
        # compute the weighted hessian matrix 
        H = - X_train.T.dot(np.diag(w * h * (1 - h))).dot(X_train) - 1e-4 * np.eye(n)
        
        # update the parameters
        theta = theta - np.linalg.solve(H, g)
    
    # Return predicted y
    return float(x.dot(theta) > 0)
