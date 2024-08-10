import numpy as np
import matplotlib.pyplot as plt

def plot_ica_filters(W, W_z):
    """
    Plot ICA filters in a grid.
    
    Args:
    W (numpy.ndarray): ICA unmixing matrix
    patch_size (int): Size of each patch (assumed to be square)
    W_z (numpy.ndarray): Whitening matrix
    """
    
    patch_size = 16
    
    # Sort filters by 2-norm
    F = W @ W_z
    norms = np.linalg.norm(F, axis=1)
    idxs = np.argsort(norms)

    # Create a big image to hold all filters
    n_filters = W.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    big_filters = np.min(W) * np.ones(((patch_size+1)*grid_size-1, (patch_size+1)*grid_size-1))

    # Plot filters in the big image
    for i in range(grid_size):
        for j in range(grid_size):
            if i*grid_size + j < n_filters:
                filter_idx = idxs[i*grid_size + j]
                filter_image = W[filter_idx, :].reshape(patch_size, patch_size)
                big_filters[i*(patch_size+1):(i+1)*(patch_size+1)-1, 
                            j*(patch_size+1):(j+1)*(patch_size+1)-1] = filter_image

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(big_filters, cmap='gray')
    plt.axis('off')
    plt.title('ICA Filters')
    plt.show()