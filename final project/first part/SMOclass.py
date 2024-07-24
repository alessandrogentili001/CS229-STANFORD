import numpy as np

''' implement SVM with SMO algorithm in the MAX MARGIN framework'''

class SVMClassification:
    def __init__(self, C=1.0, tol=1e-3, eps=1e-3, max_passes=5):
        self.C = C  # Regularization parameter
        self.tol = tol  # Numerical tolerance
        self.eps = eps  # Epsilon for comparison
        self.max_passes = max_passes  # Maximum number of passes
        self.alphas = None  # Lagrange multipliers
        self.b = 0  # Bias term
        self.X = None  # Training data
        self.y = None  # Labels
        self.error_cache = None  # Error cache
        self.m = 0  # Number of training examples
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.sign(self._decision_function(X))
        
    def train(self, X, y):
        # main loop 
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.alphas = np.zeros(self.m)
        self.error_cache = np.zeros(self.m)
        
        num_changed = 0
        examine_all = True
        
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.m):
                    num_changed += self._examine_example(i)
            else:
                for i in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
                    num_changed += self._examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
    
    def _examine_example(self, i2):
        y2 = self.y[i2]
        alph2 = self.alphas[i2]
        E2 = self._error(i2)
        r2 = E2 * y2
        
        if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):
            if np.sum((self.alphas != 0) & (self.alphas != self.C)) > 1:
                i1 = self._second_choice_heuristic(i2)
                if self._take_step(i1, i2):
                    return 1
            
            for i1 in np.random.permutation(np.where((self.alphas != 0) & (self.alphas != self.C))[0]):
                if self._take_step(i1, i2):
                    return 1
            
            for i1 in np.random.permutation(range(self.m)):
                if self._take_step(i1, i2):
                    return 1
        
        return 0
    
    def _take_step(self, i1, i2):
        if i1 == i2:
            return False
        
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self._error(i1)
        E2 = self._error(i2)
        s = y1 * y2
        
        L, H = self._compute_L_H(i1, i2)
        if L == H:
            return False
        
        k11 = self._kernel(self.X[i1], self.X[i1])
        k12 = self._kernel(self.X[i1], self.X[i2])
        k22 = self._kernel(self.X[i2], self.X[i2])
        eta = k11 + k22 - 2 * k12
        
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            Lobj = self._objective_function(L, i1, i2)
            Hobj = self._objective_function(H, i1, i2)
            if Lobj < Hobj - self.eps:
                a2 = L
            elif Lobj > Hobj + self.eps:
                a2 = H
            else:
                a2 = alph2
        
        if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return False
        
        a1 = alph1 + s * (alph2 - a2)
        
        self._update_threshold(i1, i2, a1, a2)
        self._update_error_cache(i1, i2, a1, a2)
        
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        
        return True
    
    def _error(self, i):
        return self._decision_function(self.X[i]) - self.y[i]
    
    def _decision_function(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.sum((self.alphas * self.y)[:, np.newaxis] * self._kernel(self.X, X), axis=0) + self.b

    def _kernel(self, X1, X2):
        return np.dot(X1, X2.T)  # Linear kernel
    
    def _compute_L_H(self, i1, i2):
        if self.y[i1] != self.y[i2]:
            L = max(0, self.alphas[i2] - self.alphas[i1])
            H = min(self.C, self.C + self.alphas[i2] - self.alphas[i1])
        else:
            L = max(0, self.alphas[i1] + self.alphas[i2] - self.C)
            H = min(self.C, self.alphas[i1] + self.alphas[i2])
        return L, H
    
    def _objective_function(self, a2, i1, i2):
        y1, y2 = self.y[i1], self.y[i2]
        alph1, alph2 = self.alphas[i1], self.alphas[i2]
        E1, E2 = self._error(i1), self._error(i2)
        s = y1 * y2
        
        # Calculate new alpha1
        a1 = alph1 + s * (alph2 - a2)
        
        # Calculate objective function value
        k11 = self._kernel(self.X[i1], self.X[i1])
        k12 = self._kernel(self.X[i1], self.X[i2])
        k22 = self._kernel(self.X[i2], self.X[i2])
        
        obj = a1 + a2 - 0.5 * k11 * a1 * a1 - 0.5 * k22 * a2 * a2 - s * k12 * a1 * a2 \
              - y1 * a1 * E1 - y2 * a2 * E2
        
        return obj
    
    def _second_choice_heuristic(self, i2):
        E2 = self._error(i2)
        max_delta_E = 0
        i1 = -1
        
        for i in range(self.m):
            if self.alphas[i] > 0 and self.alphas[i] < self.C:
                E1 = self._error(i)
                delta_E = abs(E1 - E2)
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    i1 = i
        
        if i1 >= 0:
            return i1
        else:
            return np.random.randint(0, self.m)
    
    def _update_threshold(self, i1, i2, a1, a2):
        y1, y2 = self.y[i1], self.y[i2]
        E1, E2 = self._error(i1), self._error(i2)
        
        b1 = self.b - E1 - y1 * (a1 - self.alphas[i1]) * self._kernel(self.X[i1], self.X[i1]) \
             - y2 * (a2 - self.alphas[i2]) * self._kernel(self.X[i1], self.X[i2])
        
        b2 = self.b - E2 - y1 * (a1 - self.alphas[i1]) * self._kernel(self.X[i1], self.X[i2]) \
             - y2 * (a2 - self.alphas[i2]) * self._kernel(self.X[i2], self.X[i2])
        
        if 0 < a1 < self.C:
            self.b = b1
        elif 0 < a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
    
    def _update_error_cache(self, i1, i2, a1, a2):
        self.error_cache[i1] = 0
        self.error_cache[i2] = 0
        
        for i in range(self.m):
            if 0 < self.alphas[i] < self.C:
                self.error_cache[i] = self._error(i)

    def _error(self, i):
        if 0 < self.alphas[i] < self.C:
            return self.error_cache[i]
        return self._decision_function(self.X[i]) - self.y[i]

