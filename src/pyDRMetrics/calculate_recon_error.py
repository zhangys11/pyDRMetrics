# 
# Calculate the reconstruction error

import numpy as np
from numpy.linalg import norm

def recon_MSE(X,Xr):
    assert X.shape == Xr.shape
    return norm(X-Xr, ord='fro')**2/(X.shape[0]*X.shape[1])


def calculate_recon_error(X, Xr):
    assert X.shape == Xr.shape

    mse = recon_MSE(X, Xr) # mean square error
    ms = recon_MSE(X, np.zeros(X.shape)) # mean square of original data matrix

    return mse, ms, mse/ms