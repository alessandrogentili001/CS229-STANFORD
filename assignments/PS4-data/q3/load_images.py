import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt 

def load_images():
    
    patch_size = 16

    X_ica = np.zeros((patch_size * patch_size, 40000))
    idx = 0

    # load images and convert to grayscale
    for i in range(1, 5):
        print(f"Loading image {i}.jpg")
        image = np.array(Image.open(os.path.join('assignments\PS4-data\q3\images', f'{i}.jpg')).convert('L'))
        
        # show image 
        # plt.imshow(image, cmap='gray')
        # plt.title(f'Image {i}')
        # plt.show()
        
        # process the image 
        y, x = image.shape
        for i in range(y // patch_size):
            for j in range(x // patch_size):
                patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] # select the next patch from the image
                X_ica[:, idx] = patch.reshape(patch_size * patch_size) # add a new column to the matrix (patch)
                idx += 1

    X_ica = X_ica[:, :idx] # remove unused columns
    W_z = np.linalg.inv(((1/X_ica.shape[1]) * X_ica @ X_ica.T) ** 0.5) # compute W_z
    X_ica = X_ica - np.mean(X_ica, axis=1, keepdims=True) # center the data
    X_pca = X_ica.copy() # define pca matrix 

    X_ica = 2 * W_z @ X_ica # apply W_z to ica data
    X_pca = X_pca / np.std(X_pca, axis=1, keepdims=True) # normalize pca data

    return X_ica, W_z, X_pca

# load_images()