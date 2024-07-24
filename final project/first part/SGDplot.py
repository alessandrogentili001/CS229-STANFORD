import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from SGDclass import SVMClassification  # Assuming the class is in a file named SVMClassification.py

def create_linearly_separable_dataset(n_samples=100, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=random_state, cluster_std=1.2)
    # Convert labels to 0 and 1
    y = (y + 1) // 2
    
    # Add some random noise
    X += np.random.normal(0, 0.3, X.shape)
    
    # Add some outliers
    n_outliers = int(0.05 * n_samples)
    outliers_x = np.random.uniform(low=-2, high=2, size=(n_outliers, 2))
    outliers_y = np.random.choice([0, 1], size=n_outliers)
    X = np.vstack([X, outliers_x])
    y = np.hstack([y, outliers_y])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def create_nonlinear_dataset(n_samples=100, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
    # y is already 0 and 1 for make_moons
    
    # Add some random noise
    X += np.random.normal(0, 0.3, X.shape)
    
    # Add some outliers
    n_outliers = int(0.05 * n_samples)
    outliers_x = np.random.uniform(low=-2, high=2, size=(n_outliers, 2))
    outliers_y = np.random.choice([0, 1], size=n_outliers)
    X = np.vstack([X, outliers_x])
    y = np.hstack([y, outliers_y])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def train_and_plot(X_train, y_train, X_test, y_test, C, ax, regularizer):
    svm = SVMClassification(C=C, learning_rate=0.01, epochs=1000, batch_size=32, regularizer=regularizer)
    svm.fit(X_train, y_train)
    
    # Calculate accuracy
    train_accuracy = svm.score(X_train, y_train)
    test_accuracy = svm.score(X_test, y_test)
    
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
    ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], s=100, 
               linewidth=1, facecolors='none', edgecolors='k')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f'{regularizer.upper()}, C = {C}')
    
    # Print some information about the model
    print(f"{regularizer.upper()}, C = {C}:")
    print(f"  Number of support vectors: {len(svm.support_vectors)}")
    print(f"  Bias term (b): {svm.b:.2f}")
    print(f"  Train accuracy: {train_accuracy:.2f}")
    print(f"  Test accuracy: {test_accuracy:.2f}")
    
    return train_accuracy, test_accuracy

# Create datasets
X_linear, y_linear = create_linearly_separable_dataset(n_samples=100, random_state=42)
X_nonlinear, y_nonlinear = create_nonlinear_dataset(n_samples=100, random_state=42)

# Split datasets into train and test sets
X_linear_train, X_linear_test, y_linear_train, y_linear_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
X_nonlinear_train, X_nonlinear_test, y_nonlinear_train, y_nonlinear_test = train_test_split(X_nonlinear, y_nonlinear, test_size=0.2, random_state=42)

# Train and plot for different C values and regularizers
C_values = [0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0]
regularizers = ['l1', 'l2', 'elastic']

results = {reg: {'linear': {'train': [], 'test': []}, 'nonlinear': {'train': [], 'test': []}} for reg in regularizers}

for i, regularizer in enumerate(regularizers):
    
    # Set up the plot
    fig, axs = plt.subplots(2, len(C_values), figsize=(20, 15))
    
    for j, C in enumerate(C_values):
        print(f"\nLinearly Separable Dataset ({regularizer}):")
        train_acc, test_acc = train_and_plot(X_linear_train, y_linear_train, X_linear_test, y_linear_test, C, axs[0, j], regularizer)
        results[regularizer]['linear']['train'].append(train_acc)
        results[regularizer]['linear']['test'].append(test_acc)
        
        print(f"\nNon-linearly Separable Dataset ({regularizer}):")
        train_acc, test_acc = train_and_plot(X_nonlinear_train, y_nonlinear_train, X_nonlinear_test, y_nonlinear_test, C, axs[1, j], regularizer)
        results[regularizer]['nonlinear']['train'].append(train_acc)
        results[regularizer]['nonlinear']['test'].append(test_acc)
    
    axs[0, 0].set_ylabel('Linearly Separable')
    axs[1, 0].set_ylabel('Non-linearly Separable')
    plt.suptitle(f'{regularizer.upper()} Regularizer')
    plt.tight_layout()
    plt.show()

# Plot accuracy comparison
plt.figure(figsize=(12, 8))
colors = {'l1': 'blue', 'l2': 'red', 'elastic': 'green'}
linestyles = {'train': '-', 'test': '--'}

for regularizer in regularizers:
    for dataset in ['linear', 'nonlinear']:
        for acc_type in ['train', 'test']:
            plt.plot(C_values, results[regularizer][dataset][acc_type], 
                     color=colors[regularizer], 
                     linestyle=linestyles[acc_type], 
                     marker='o',
                     label=f'{regularizer.upper()} {dataset.capitalize()} {acc_type.capitalize()}')

plt.xscale('log')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy Comparison for Different Regularizers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.show()