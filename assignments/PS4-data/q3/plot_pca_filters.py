import numpy as np
import matplotlib.pyplot as plt

def plot_pca_filters(U):
    
    patch_size = 16
    
    # Calculate the size of the big image
    big_image_size = (patch_size + 1) * patch_size - 1
    
    # Create the big image filled with the minimum value from U
    big_filters = np.min(U) * np.ones((big_image_size, big_image_size))
    
    for i in range(patch_size):
        for j in range(patch_size):
            # Calculate the correct indices for placing each filter
            row_start = i * (patch_size + 1)
            row_end = row_start + patch_size
            col_start = j * (patch_size + 1)
            col_end = col_start + patch_size
            
            # Extract the filter and reshape it
            filter_patch = U[:, i * patch_size + j].reshape(patch_size, patch_size)
            
            # Place the filter in the big image
            big_filters[row_start:row_end, col_start:col_end] = filter_patch

    plt.imshow(big_filters, cmap='gray')
    plt.axis('square')
    plt.axis('off')
    plt.show()