import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from SMOclass import SVMClassification  # Make sure this matches your filename
from sklearn.model_selection import train_test_split

def create_linearly_separable_dataset(n_samples=100, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=random_state, cluster_std=1.2)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1
    
    # Add some random noise
    X += np.random.normal(0, 0.3, X.shape)
    
    # Add some outliers
    n_outliers = int(0.05 * n_samples)
    outliers_x = np.random.uniform(low=-2, high=2, size=(n_outliers, 2))
    outliers_y = np.random.choice([-1, 1], size=n_outliers)
    X = np.vstack([X, outliers_x])
    y = np.hstack([y, outliers_y])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def create_nonlinear_dataset(n_samples=100, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1
    
    # Add some random noise
    X += np.random.normal(0, 0.3, X.shape)
    
    # Add some outliers
    n_outliers = int(0.05 * n_samples)
    outliers_x = np.random.uniform(low=-2, high=2, size=(n_outliers, 2))
    outliers_y = np.random.choice([-1, 1], size=n_outliers)
    X = np.vstack([X, outliers_x])
    y = np.hstack([y, outliers_y])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def calculate_accuracy(y_true, y_pred):
    return round(np.mean(y_true == y_pred), 2)

def train_and_plot(X_train, y_train, X_test, y_test, C, ax):
    svm = SVMClassification(C=C, tol=1e-3, eps=1e-3, max_passes=100)
    svm.train(X_train, y_train)
    
    # Calculate accuracy
    train_accuracy = calculate_accuracy(y_train, svm.predict(X_train))
    test_accuracy = calculate_accuracy(y_test, svm.predict(X_test))
    
    # Create a mesh to plot in
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict the function value for the whole grid
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
    
    # Plot support vectors
    support_vectors = X_train[np.where((svm.alphas > 1e-5) & (svm.alphas < svm.C))[0]]
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, 
               linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    ax.set_title(f'C = {C}')
    
    # Print some information about the model
    n_sv = np.sum((svm.alphas > 1e-5) & (svm.alphas < svm.C))
    print(f"C = {C}:")
    print(f"  Number of support vectors: {n_sv}")
    print(f"  Bias term (b): {svm.b}")
    print(f"  Train accuracy: {train_accuracy}")
    print(f"  Test accuracy: {test_accuracy}")
    
    return train_accuracy, test_accuracy

# Create datasets
X_linear, y_linear = create_linearly_separable_dataset(n_samples=100, random_state=42)
X_nonlinear, y_nonlinear = create_nonlinear_dataset(n_samples=100, random_state=42)

# Split datasets into train and test sets
X_linear_train, X_linear_test, y_linear_train, y_linear_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
X_nonlinear_train, X_nonlinear_test, y_nonlinear_train, y_nonlinear_test = train_test_split(X_nonlinear, y_nonlinear, test_size=0.2, random_state=42)

# Train and plot for different C values
C_values = [0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0]
linear_train_acc = []
linear_test_acc = []
nonlinear_train_acc = []
nonlinear_test_acc = []

# Set up the plot
fig, axs = plt.subplots(2, len(C_values), figsize=(20, 10))

for i, C in enumerate(C_values):
    print("\nLinearly Separable Dataset:")
    train_acc, test_acc = train_and_plot(X_linear_train, y_linear_train, X_linear_test, y_linear_test, C, axs[0, i])
    linear_train_acc.append(train_acc)
    linear_test_acc.append(test_acc)
    
    print("\nNon-linearly Separable Dataset:")
    train_acc, test_acc = train_and_plot(X_nonlinear_train, y_nonlinear_train, X_nonlinear_test, y_nonlinear_test, C, axs[1, i])
    nonlinear_train_acc.append(train_acc)
    nonlinear_test_acc.append(test_acc)

axs[0, 0].set_ylabel('Linearly Separable')
axs[1, 0].set_ylabel('Non-linearly Separable')
plt.tight_layout()
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.plot(C_values, linear_train_acc, 'bo-', label='Linear Train')
plt.plot(C_values, linear_test_acc, 'b--', label='Linear Test')
plt.plot(C_values, nonlinear_train_acc, 'ro-', label='Non-linear Train')
plt.plot(C_values, nonlinear_test_acc, 'r--', label='Non-linear Test')
plt.xscale('log')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.ylim(.5, 1.0) 
plt.show()

