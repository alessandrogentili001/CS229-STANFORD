import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from l1ls import l1ls  # Assuming we've saved our function in l1ls.py

# Load the data
X, y, true_theta = load_data(r'assignments\PS3-data\q3\data\x.dat', 
                        r'assignments\PS3-data\q3\data\y.dat', 
                        r'assignments\PS3-data\q3\data\theta.dat')

# Define lambda range
lambda_range = np.logspace(-3, 1, 20)  # 20 points from 10^-3 to 10^1

# Store results
thetas = []

# Run l1ls for each lambda
for lambda_val in lambda_range:
    theta = l1ls(X, y, lambda_val)
    thetas.append(theta)

# Convert to numpy array for easier manipulation
thetas = np.array(thetas)

# Plot results
plt.figure(figsize=(12, 10))

# Subplot 1: Feature Coefficients vs Lambda
plt.subplot(2, 1, 1)
for i in range(thetas.shape[1]):
    plt.semilogx(lambda_range, thetas[:, i], label=f'Feature {i+1}')
plt.xlabel('Lambda')
plt.ylabel('Coefficient Value')
plt.title('Feature Coefficients vs Lambda')
plt.legend()
plt.grid(True)

# Subplot 2: Comparison with True Theta
plt.subplot(2, 1, 2)
mse_values = np.mean((thetas - true_theta)**2, axis=1)
plt.semilogx(lambda_range, mse_values)
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.title('MSE between Estimated Theta and True Theta')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print number of non-zero coefficients for each lambda
for i, lambda_val in enumerate(lambda_range):
    non_zero = np.sum(np.abs(thetas[i]) > 1e-5)
    print(f"Lambda = {lambda_val:.3f}: {non_zero} non-zero coefficients")

# Print the true theta
print("\nTrue Theta:")
print(true_theta)

# Print the estimated theta for the lambda with lowest MSE
best_lambda_index = np.argmin(mse_values)
best_lambda = lambda_range[best_lambda_index]
best_theta = thetas[best_lambda_index]
print(f"\nBest Lambda: {best_lambda:.3f}")
print("Estimated Theta for Best Lambda:", best_theta)