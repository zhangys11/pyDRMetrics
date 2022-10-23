# 
# Calculate the reconstruction error

import numpy as np

def recon_MSE(X,Xr):
    '''
    Reconstruction relateive MSE
    '''
    assert X.shape == Xr.shape
    return np.linalg.norm(X-Xr, ord='fro')**2/(X.shape[0]*X.shape[1])

def calculate_recon_error(X, Xr):
    '''
    Recontruction errors

    Return
    ------
    mse : mean squared error between X and Xr
    ams : mean square of X
    mse/ams : relative mean squared error
    '''
    assert X.shape == Xr.shape

    mse = recon_MSE(X, Xr) # mean square error
    ams = recon_MSE(X, np.zeros(X.shape)) # mean square of original data matrix

    return mse, ams, mse/ams

def recon_rMSE(X, Xr):
    '''
    relative mean squared error
    '''
    return recon_MSE(X, Xr) / recon_MSE(X, np.zeros(X.shape))