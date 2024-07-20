import pandas as pd
import numpy as np
from load_data import parse_sparse_arff
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 

# Define the data directories
test_file_path = r'assignments\PS2-data\q4\data\spam_test.arff'
train_file_paths = [
    r'assignments\PS2-data\q4\data\spam_train_10.arff',
    r'assignments\PS2-data\q4\data\spam_train_25.arff',
    r'assignments\PS2-data\q4\data\spam_train_50.arff',
    r'assignments\PS2-data\q4\data\spam_train_100.arff',
    r'assignments\PS2-data\q4\data\spam_train_200.arff',
    r'assignments\PS2-data\q4\data\spam_train_300.arff',
    r'assignments\PS2-data\q4\data\spam_train_400.arff',
    r'assignments\PS2-data\q4\data\spam_train_500.arff',
    r'assignments\PS2-data\q4\data\spam_train_750.arff',
    r'assignments\PS2-data\q4\data\spam_train_1000.arff',
    r'assignments\PS2-data\q4\data\spam_train_1250.arff',
    r'assignments\PS2-data\q4\data\spam_train_1500.arff',
    r'assignments\PS2-data\q4\data\spam_train_1750.arff',
    r'assignments\PS2-data\q4\data\spam_train_2000.arff',
]

# Load test data
Xtest, ytest, feature_names = parse_sparse_arff(test_file_path)
Xtest = csr_matrix(Xtest)
ytest = np.array(ytest)

print(f"Shape of Xtest: {Xtest.shape}")
print(f"Shape of ytest: {ytest.shape}")

# Initialize results dictionary
results = {
    'train_size': [],
    'svm_accuracy': [], 'svm_precision': [], 'svm_recall': [], 'svm_f1': [],
    'nb_accuracy': [], 'nb_precision': [], 'nb_recall': [], 'nb_f1': []
}

# Loop over all train data
for train_file_path in train_file_paths:
    Xtrain, ytrain, _ = parse_sparse_arff(train_file_path)
    Xtrain = csr_matrix(Xtrain)
    ytrain = np.array(ytrain)
    
    print(f"\nTraining on: {train_file_path}")
    print(f"Shape of Xtrain: {Xtrain.shape}")
    print(f"Shape of ytrain: {ytrain.shape}")
    
    # Train SVM model
    svm = SVC(kernel='linear')
    svm.fit(Xtrain, ytrain)
    svm_pred = svm.predict(Xtest)
    
    # Train Naive Bayes model
    nb = MultinomialNB()
    nb.fit(Xtrain, ytrain)
    nb_pred = nb.predict(Xtest)
    
    # Calculate metrics
    results['train_size'].append(Xtrain.shape[0])
    
    for name, pred in [('svm', svm_pred), ('nb', nb_pred)]:
        results[f'{name}_accuracy'].append(accuracy_score(ytest, pred))
        results[f'{name}_precision'].append(precision_score(ytest, pred, average='weighted'))
        results[f'{name}_recall'].append(recall_score(ytest, pred, average='weighted'))
        results[f'{name}_f1'].append(f1_score(ytest, pred, average='weighted'))
    
    print(f"SVM - Accuracy: {results['svm_accuracy'][-1]:.4f}, F1: {results['svm_f1'][-1]:.4f}")
    print(f"NB - Accuracy: {results['nb_accuracy'][-1]:.4f}, F1: {results['nb_f1'][-1]:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df)

# You can save the results to a CSV file if needed
# results_df.to_csv('spam_classification_results.csv', index=False)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison: SVM vs Naive Bayes', fontsize=16)

# List of metrics to plot
metrics = ['accuracy', 'precision', 'recall', 'f1']

# Plot each metric
for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    
    # Plot SVM performance
    sns.lineplot(x='train_size', y=f'svm_{metric}', data=results_df, ax=ax, label='SVM')
    
    # Plot Naive Bayes performance
    sns.lineplot(x='train_size', y=f'nb_{metric}', data=results_df, ax=ax, label='Naive Bayes')
    
    ax.set_title(f'{metric.capitalize()} vs Training Set Size')
    ax.set_xlabel('Number of Training Examples')
    ax.set_ylabel(metric.capitalize())
    ax.legend()

    # Set x-axis to log scale for better visualization
    ax.set_xscale('log')
    
    # Add gridlines
    ax.grid(True, which="both", ls="-", alpha=0.2)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# Optionally, save the figure
# plt.savefig('spam_classification_performance.png', dpi=300, bbox_inches='tight')
