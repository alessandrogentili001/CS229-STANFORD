from load_data import load_data
from lwlr import lwlr
from plot_lwlr import plot_lwlr

# Load the data
X, y = load_data(r'assignments\PS1-data\q2\data\x.dat', r'assignments\PS1-data\q2\data\y.dat')

# Define tau values to test
tau_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# Plot decision boundaries for each tau
for i, tau in enumerate(tau_values):
    plot_lwlr(X, y, tau, res=100, lwlr=lwlr)