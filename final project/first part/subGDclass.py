import numpy as np

class SVMClassification:
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000, batch_size=32, regularizer='l2', l1_ratio=0.5):
        self.C = C  # Regularization parameter
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularizer = regularizer  # 'l1', 'l2', or 'elastic'
        self.l1_ratio = l1_ratio  # Used only for elastic net (mix of L1 and L2)
        self.w = None  # Weight vector
        self.b = 0  # Bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for _ in range(self.epochs):
            for _ in range(0, n_samples, self.batch_size):
                batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Compute subgradients
                margin = y_batch * (np.dot(X_batch, self.w) + self.b)
                misclassified = margin < 1
                
                dw, db = self._compute_subgradients(X_batch, y_batch, misclassified)
                
                # Update parameters using subgradient descent
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            # Optional: Decay learning rate
            self.learning_rate *= 0.99

    def _compute_subgradients(self, X_batch, y_batch, misclassified):
        dw = self._compute_weight_subgradient(X_batch, y_batch, misclassified)
        db = -self.C * np.sum(y_batch[misclassified])
        return dw, db

    def _compute_weight_subgradient(self, X_batch, y_batch, misclassified):
        subgradient = -self.C * np.sum(y_batch[misclassified][:, np.newaxis] * X_batch[misclassified], axis=0)
        
        if self.regularizer == 'l2':
            subgradient += self.w
        elif self.regularizer == 'l1':
            subgradient += np.sign(self.w)
        elif self.regularizer == 'elastic':
            subgradient += self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * self.w
        
        return subgradient

    def predict(self, X):
        return np.where(self._decision_function(X) > 0, 1, -1)

    def _decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def hinge_loss(self, X, y):
        margin = y * self._decision_function(X)
        loss = np.maximum(0, 1 - margin)
        return np.mean(loss) + self._regularization_term()

    def _regularization_term(self):
        if self.regularizer == 'l2':
            return 0.5 * np.sum(self.w ** 2)
        elif self.regularizer == 'l1':
            return np.sum(np.abs(self.w))
        elif self.regularizer == 'elastic':
            return (self.l1_ratio * np.sum(np.abs(self.w)) +
                    (1 - self.l1_ratio) * 0.5 * np.sum(self.w ** 2))
        else:
            return 0