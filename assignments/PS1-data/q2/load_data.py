import numpy as np

def load_data(x_file, y_file):
    X = np.loadtxt(x_file)
    y = np.loadtxt(y_file)
    return X, y

# Usage
X, y = load_data(r'assignments\PS1-data\q2\data\x.dat', r'assignments\PS1-data\q2\data\y.dat')

print("shape of X:", X.shape)
print("shape of y:", y.shape)