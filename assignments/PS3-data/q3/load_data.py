import numpy as np

def load_data(X_dir, y_dir, theta_dir):
    # Load X data
    X = np.loadtxt(X_dir)
    
    # Load y data
    y = np.loadtxt(y_dir)
    
    # Load theta data
    theta = np.loadtxt(theta_dir)
    
    return X, y, theta

# show data 
X, y, theta = load_data(r'assignments\PS3-data\q3\data\x.dat', 
                        r'assignments\PS3-data\q3\data\y.dat', 
                        r'assignments\PS3-data\q3\data\theta.dat')

print("X: ", X, "shape: ", X.shape)
print("y: ", y, "shape: ", y.shape)
print("theta: ", theta, "shape: ", theta.shape)