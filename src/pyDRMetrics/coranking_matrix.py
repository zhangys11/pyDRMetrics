import math
import numpy as np
from tqdm import tqdm


def ranking_matrix(D, solver='fast'):    
    '''
    Get the ranking matrix from distance matrix

    Parameters
    ----------
    D - distance matrix
    solver - default 'fast'. 

    Return
    ------
    R - ranking matrix
    '''

    D = np.array(D)    

    if solver == 'fast':
        # Computing ranking_matrix using np.argsort twice on every row. By Nilrad
        R = [np.argsort(np.argsort(row)) for row in D] 
    else:    
        R = np.zeros(D.shape)
        m = len(R)
        
        for i in range(m):
            for j in range(m):
                Rij = 0
                for k in range(m):
                    if (D[i,k] < D[i,j]) or (math.isclose(D[i,k], D[i,j]) and k < j ):
                        Rij += 1
                R[i,j] = Rij

    return np.array(R, dtype = 'uint')


def coranking_matrix(R1, R2):
    '''
    Get co-ranking matrix

    Parameters
    ----------
    R1, R2 - two ranking matrices as input

    Return
    ------
    Q - coranking matrix
    '''


    R1 = np.array(R1)
    R2 = np.array(R2)    
    assert R1.shape == R2.shape    
    Q = np.zeros(R1.shape)
    m = len(Q)

    #for k in tqdm(range(m)): # Constructing Q is the most time-consuming process
    #    for l in range(m):
    #        kl = 0
    #        for i in range(m):
    #            for j in range(m):
    #                if R1[i,j] == k and R2[i,j] == l:
    #                    kl += 1
    #        Q[k,l] = kl

    for i in range(m):
        for j in range(m):
            k = int(R1[i,j])
            l = int(R2[i,j])
            Q[k,l] += 1

    return Q


def coranking_matrix_metrics(Q):
    
    Q = Q[1:,1:]
    m = len(Q)
    
    T = np.zeros(m-1) # trustworthiness
    C = np.zeros(m-1) # continuity
    QNN = np.zeros(m) # Co-k-nearest neighbor size
    LCMC = np.zeros(m) # Local Continuity Meta Criterion
    
    for k in range(m-1):
        Qs = Q[k:,:k]
        W = np.arange(Qs.shape[0]).reshape(-1, 1) # a column vector of weights. weight = rank error = actual_rank - k
        T[k] = 1-np.sum(Qs * W)/(k+1)/m/(m-1-k)  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
        Qs = Q[:k,k:]
        W = np.arange(Qs.shape[1]).reshape(1, -1) # a row vector of weights. weight = rank error = actual_rank - k
        C[k] = 1-np.sum(Qs * W)/(k+1)/m/(m-1-k)  # 1 - normalized hard-k-extrusions. upper-right region 
    
    for k in range(m):    
        QNN[k] = np.sum(Q[:k+1,:k+1])/((k+1) * m) # Q[0,0] is always m. 0-th nearest neighbor is always the point itself. Exclude Q[0,0]
        LCMC[k] = QNN[k] - (k+1)/(m-1)

    kmax = np.argmax(LCMC)
    Qlocal = np.sum(QNN[:kmax+1])/(kmax + 1)
    Qglobal = np.sum(QNN[kmax:-1])/(m - kmax -1) # skip the last. The last is (m-1)-nearest neighbor, including all samples.
    AUC = np.mean(QNN)
    
    return T, C, QNN, AUC, LCMC, kmax, Qlocal, Qglobal