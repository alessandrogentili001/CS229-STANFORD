import numpy as np
import matplotlib.pyplot as plt 

def load_data(X_dir):
    X = np.loadtxt(X_dir)
    return X

# show data
X = load_data(r'assignments\PS3-data\q4\X.dat')
print("data shape: ", X.shape)

# plot data 
plt.scatter(X[:, 0], X[:, 1])
plt.title('Original Data')
plt.show()

