#
# Sample Reconstruction Visualization: Compare orginal and reconstructed samples side by side

from matplotlib import pyplot as plt
from .plt2base64 import *

def visualize_sample_reconstruction(X, Xr, X_names, N = 3, silent = False):
    
    assert X.shape == Xr.shape
    assert X.shape[1] == len(X_names)

    # number of test samples
    N = min(N, X.shape[0])
    
    plt.figure(figsize=(18, 2*N)) # 100, 12*N

    if (silent == False):
        print("left column: original waveform\t\t right column: recovered waveform")

    for i in range(N):
        # original
        ax = plt.subplot(N, 2, i*2 + 1)
        plt.scatter(X_names, list(X[i]), s = 1)        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstruction
        ax = plt.subplot(N, 2, i*2 + 2)
        plt.scatter(X_names, list(Xr[i]), s = 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    s = plt2html(plt)

    if (silent == False):
        plt.show()

    return s