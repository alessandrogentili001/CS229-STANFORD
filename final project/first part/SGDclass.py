import numpy as np

class SVMClassification:
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000, batch_size=32, regularizer='l2', l1_ratio=0.5):
        self.C = C  # Regularization parameter
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularizer = regularizer  # 'l1', 'l2', or 'elastic'
        self.l1_ratio = l1_ratio  # Used only for elastic net (mix of L1 and L2)
        self.alpha = None  # Dual coefficients
        self.support_vectors = None  # Support vectors
        self.b = 0  # Bias term

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        
        # Convert 0/1 labels to -1/1
        y = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):
            for _ in range(0, n_samples, self.batch_size):
                batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                alpha_batch = self.alpha[batch_indices]
                
                # Compute gradients
                kernel_matrix = self.linear_kernel(X_batch, X)
                margin = y_batch * (np.dot(kernel_matrix, self.alpha * y) + self.b)
                misclassified = margin < 1
                
                dalpha = self._compute_gradient(kernel_matrix, y_batch, misclassified, alpha_batch)
                db = -self.C * np.sum(y_batch[misclassified])
                
                # Update parameters
                self.alpha[batch_indices] -= self.learning_rate * dalpha
                self.b -= self.learning_rate * db

        # Store support vectors
        support_vector_indices = np.where(np.abs(self.alpha) > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]
        self.alpha = self.alpha[support_vector_indices]
        self.y = y[support_vector_indices]

    def _compute_gradient(self, kernel_matrix, y_batch, misclassified, alpha_batch):
        gradient = -self.C * y_batch * misclassified
        
        if self.regularizer == 'l2':
            gradient += alpha_batch
        elif self.regularizer == 'l1':
            gradient += np.sign(alpha_batch)
        elif self.regularizer == 'elastic':
            gradient += self.l1_ratio * np.sign(alpha_batch) + (1 - self.l1_ratio) * alpha_batch
        
        return gradient

    def predict(self, X):
        return np.where(self._decision_function(X) > 0, 1, 0)

    def _decision_function(self, X):
        kernel_matrix = self.linear_kernel(X, self.support_vectors)
        return np.dot(kernel_matrix, self.alpha * self.y) + self.b

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def hinge_loss(self, X, y):
        margin = y * self._decision_function(X)
        loss = np.maximum(0, 1 - margin)
        return np.mean(loss) + self._regularization_term()

    def _regularization_term(self):
        if self.regularizer == 'l2':
            return 0.5 * np.sum(self.alpha ** 2)
        elif self.regularizer == 'l1':
            return np.sum(np.abs(self.alpha))
        elif self.regularizer == 'elastic':
            return (self.l1_ratio * np.sum(np.abs(self.alpha)) +
                    (1 - self.l1_ratio) * 0.5 * np.sum(self.alpha ** 2))
        else:
            return 0