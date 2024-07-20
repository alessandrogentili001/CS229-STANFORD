import numpy as np
import matplotlib.pyplot as plt

def plot_lwlr(X, y, tau, res, lwlr):
    
    # initialize the x and the predicted values
    x = np.zeros(2)
    pred = np.zeros((res, res))
    
    # for loop to compute the predicted values
    for i in range(res):
        for j in range(res):
            x[0] = 2 * i / (res - 1) - 1
            x[1] = 2 * j / (res - 1) - 1
            pred[j, i] = lwlr(X, y, x, tau)
    
    # plot the predicted values
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(pred, extent=[-1, 1, -1, 1], origin='lower', vmin=-0.4, vmax=1.3)
    
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'ko', label='Class 0')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'kx', label='Class 1')
    
    plt.axis('equal')
    plt.title(f'tau = {tau}', fontsize=18)
    plt.legend()
    plt.show()