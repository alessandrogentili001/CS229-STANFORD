from pca import pca
from ica import ica 
from load_images import load_images
from plot_pca_filters import plot_pca_filters
from plot_ica_filters import plot_ica_filters

# load images 
print("Loading images...")
X_ica, W_z, X_pca = load_images()

# For PCA
print("Performing PCA...")
pca_components = pca(X_pca)

# For ICA
print("Performing ICA...")
ica_unmixing = ica(X_ica)

# Plot PCA filters
print("Plotting PCA filters...")
plot_pca_filters(pca_components)

# Plot ICA filters
print("Plotting ICA filters...")
plot_ica_filters(ica_unmixing, W_z)
