import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def parse_sparse_arff(file_path):
    attributes = []
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.lower().startswith('@attribute'):
                attr_name = line.split()[1]
                attributes.append(attr_name)
            elif line.startswith('{'):
                instance = np.zeros(len(attributes) - 1)  # -1 because the last attribute is the class
                values = line[1:-1].split(',')
                for value in values[:-1]:  # Last value is the class
                    index, val = map(int, value.split())
                    instance[index] = val
                data.append((instance, values[-1].split()[-1]))

    X = csr_matrix(np.array([d[0] for d in data]))
    y = np.array([d[1] for d in data])
    feature_names = attributes[:-1]  # Last attribute is the class
    return X, y, feature_names


#### EXAMPLE USAGE ####

# Load the data 
test_file_path = r'assignments\PS2-data\q4\data\spam_test.arff'
train_file_path = r'assignments\PS2-data\q4\data\spam_train_750.arff'
X_train, y_train, feature_names = parse_sparse_arff(train_file_path)
X_test, y_test, _ = parse_sparse_arff(test_file_path)

# Convert to DataFrame for easier handling
df_X_train = pd.DataFrame.sparse.from_spmatrix(X_train, columns=feature_names)
df_y_train = pd.Series(y_train, name='class')
df_X_test = pd.DataFrame.sparse.from_spmatrix(X_test, columns=feature_names)
df_y_test = pd.Series(y_test, name='class')

print(f"Shape of X_train: {df_X_train.shape}")
print(f"Shape of y_train: {df_y_train.shape}")
print(f"Shape of X_test: {df_X_test.shape}")
print(f"Shape of y_test: {df_y_test.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Classes: {np.unique(y_train).tolist()}")

# Display first few rows
print("\nFirst few rows of the dataset:")
print(df_X_train.head())

# Display distribution of classes
print("\nDistribution of classes:")
print(df_y_train.value_counts(normalize=True))

# Save to CSV if needed
# df_X.join(df_y).to_csv('spam_data.csv', index=False)