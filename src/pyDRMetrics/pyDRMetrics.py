import numpy as np
import matplotlib.pyplot as plt
from .calculate_recon_error import *
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr
from .visualize_matrix import *
from .visualize_sample_reconstruction import *
from .coranking_matrix import *
from .plt2base64 import *
import json
import os
import csv

class DRMetrics:
    '''
    Define a set of dimensionality reduction metrics.
    '''
    
    def __init__(self, X, Z, Xr = None):
        '''
        Initialization. X is the original data. Z is the data after DR. Xr is the reconstructed data.
        '''
        self.X = X
        '''Data before DR. m-by-n matrix'''
        
        self.Z = Z 
        '''Data after DR. m-by-k matrix. Typically, k << n'''
        
        # print(X.shape, Z.shape)
        assert X.shape[0] == Z.shape[0]
                
        if (Xr is not None):
            assert X.shape == Xr.shape
            self.Xr = Xr # reconstructed data after DR. m-by-n matrix
            self.mse, self.ms, self.rmse = calculate_recon_error(X, Xr)
            
        # Construct the distance matrix
        df = pd.DataFrame(X, index=None)
        self.D = pd.DataFrame(pairwise_distances(df.values)).values        

        dfz = pd.DataFrame(Z, index=None)
        self.Dz = pd.DataFrame(pairwise_distances(dfz.values)).values
        
        # Residual Variance of the two distance matrices 
        self.Vr = 1 - (pearsonr(self.D.flatten(), self.Dz.flatten())[0])**2 # Pearson's r version
        self.Vrs = 1 - (spearmanr(self.D.flatten(), self.Dz.flatten())[0])**2 # Spearman' r version

        self.R = ranking_matrix(self.D)
        self.Rz = ranking_matrix(self.Dz)
        self.Q = coranking_matrix(self.R, self.Rz)
      
        self.T, self.C, self.QNN, self.AUC, self.LCMC, self.kmax, self.Qlocal, self.Qglobal = coranking_matrix_metrics(self.Q)

        self.AUC_T = np.mean(self.T)
        self.AUC_C = np.mean(self.C)
    
    def has_header(fn):
        '''
        Guess whether a csv file has header
        '''
        has_hdr = False
        if os.path.isfile(fn):
            with open(fn, 'r') as csvfile:
                has_hdr = csv.Sniffer().has_header(csvfile.read(2048))
                csvfile.seek(0)
        return has_hdr

    @classmethod
    def from_files(cls, csv1, csv2, csv3):
        '''
        csv1 - X, csv2 - Z, csv3 - Xr. These csv file have no header.
        '''

        if (not os.path.isfile(csv1) or not os.path.isfile(csv2)):
            raise Exception("file doesnot exist")

        X = pd.read_csv(csv1, header = None).values
        Z = pd.read_csv(csv2, header = None).values
        Xr = None
        if os.path.isfile(csv3):
            Xr = pd.read_csv(csv3, header = None).values

        return cls(X, Z, Xr)

    @classmethod
    def test(cls, csv, K = 3, dr = 'PCA'):
        '''
        csv - A csv file without header.
        '''

        from sklearn.decomposition import PCA, NMF
        from sklearn.random_projection import GaussianRandomProjection
        from sklearn.manifold import MDS, TSNE
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import OneHotEncoder
        
        X = pd.read_csv(csv, header = None).values
        Z = None
        Xr = None
        
        if (dr == 'PCA'):
            pca = PCA(n_components = K) # keep the first K components
            pca.fit(X)
            Z = pca.transform(X)
            Xr = pca.inverse_transform(Z)
        elif (dr == 'NMF'):
            # make sure X is non-negative
            Xmin = np.min(X)
            if (Xmin < 0):
                X = X - Xmin

            nmf = NMF(n_components = K) # keep the first K components
            nmf.fit(X)
            Z = nmf.transform(X)
            Xr = nmf.inverse_transform(Z)

            if (Xmin < 0):
                Xr = Xr + Xmin
        elif (dr == 'RP'):
            grp = GaussianRandomProjection(n_components = K) # keep the first K components
            Z = grp.fit_transform(X)
        elif (dr == 'VQ'):            
            kmeans = KMeans(n_clusters = K).fit(X)
            Xvq = kmeans.predict(X)
            H = kmeans.cluster_centers_            
            ohe = OneHotEncoder()
            Z = ohe.fit_transform(Xvq.reshape(-1, 1)).A
            Xr = Z @ H
        elif (dr == 'MDS'):
            mds = MDS(n_components = K) # keep the first K components
            Z = mds.fit_transform(X)
        elif (dr == 'TSNE'):
            tsne = MDS(n_components = K) # keep the first K components
            Z = tsne.fit_transform(X)
        elif (dr == 'IDENTITY'):
            # for this case, k is not used.
            Z = X
            Xr = X
        else:
            raise Exception("Invalid DR name")
            
        return cls(X, Z, Xr)

    def visualize_reconstruction(self):
        if (self.Xr is not None):
            return visualize_sample_reconstruction(self.X, self.Xr, X_names=list(range(self.X.shape[1])), N = 3)
        else:
            return None

    def plot_JL_curve(self):
        '''
        Plot the Johnson-Lindenstrauss minimum dimensions curve against the maximum distortion rate for Random Projection.
        The plot is also saved to a local jpg file.
        '''
        fig = plt.figure(figsize=(6,4))        
        eps_range = np.linspace(0.01, 0.99, 100)
        min_n_components = johnson_lindenstrauss_min_dim(n_samples = len(self.X), eps=eps_range)
        plt.plot(eps_range, min_n_components)
        plt.xlabel('maximum distortion rate',fontsize=16)
        plt.ylabel('mimimum dimensions to keep',fontsize=16)
        plt.ylim(0, 20000)
        plt.title('johnson_lindenstrauss_min_dim vs max_distortion_rate \nsample size = ' + str(len(self.X)),fontsize=16)
        plt.show()
        return plt2base64(plt)
    
    def plot_distance_matrix(self):
        '''
        Plot the distance matrices before and after DR, i.e., D and Dz.
        '''
        
        visualize_matrix(self.D)
        visualize_matrix(self.Dz)
        
    def plot_ranking_matrix(self):
        '''
        Plot the ranking matrices before and after DR, i.e., R and Rz.
        '''
        
        visualize_matrix(self.R)
        visualize_matrix(self.Rz)
        
    def plot_coranking_matrix(self):
        '''
        Plot the coranking matrix between R and Rz.
        '''
        
        visualize_matrix(self.Q, cmap = 'gray_r')
        
    def report(self):
        '''
        Generate a summary report
        '''
        
        if hasattr(self, 'Xr') and self.Xr is not None:
            print("--- Sample Reconstruction (X and Xr) ---")
            self.visualize_reconstruction()
            print("rMSE = ", self.rmse)
        
        print("--- Distance Matrices (D and Dz) ---")
        self.plot_distance_matrix()
        print("Residual Variance (using Pearson's r) = ", self.Vr)
        print("Residual Variance (using Spearman's r) = ", self.Vrs)
        
        print("--- Ranking Matrices (R and Rz) ---")
        self.plot_ranking_matrix()
        
        print("--- Co-ranking Matrix (Q) ---")
        self.plot_coranking_matrix()
        
        print("--- Trustworthiness T(k) and Continuity C(k) ---")
        plt.figure()
        plt.plot(list(range(1, len(self.T) + 1)), self.T, label = 'trustworthiness')
        plt.plot(list(range(1, len(self.C) + 1)), self.C, label = 'continuity')
        plt.ylim(-0.05,1.05)
        plt.legend()
        plt.show()

        print("AUC of T = ", self.AUC_T)
        print("AUC of C = ", self.AUC_C)
        
        print("--- QNN(k) Curve ---")
        plt.figure()
        plt.ylim(-0.05,1.1)
        plt.plot(list(range(1, len(self.QNN) + 1)), self.QNN)
        plt.ylim(-0.05, 1.05)
        # plt.xlim(1, len(self.QNN) + 1)
        plt.show()
        
        print("AUC of QNN = ", self.AUC)
        
        print("--- LCMC(k) Curve ---")
        plt.figure()
        plt.ylim(-0.05,1.1)
        plt.plot(list(range(1, len(self.LCMC) + 1)), self.LCMC)
        plt.show()
        
        print("kmax (0-based index) = ", self.kmax)
        print("Qlocal = ", self.Qlocal)
        print("Qglobal = ", self.Qglobal)
        
        #print("--- Properties ---")
        #print(self.__dict__)

    def get_html(self):
        '''
        Generate a summary report in HTML format
        '''
        
        html = '<table class="table table-striped">'

        if hasattr(self, 'Xr') and self.Xr is not None:
            tr = '<tr><td>Sample Reconstruction (X and Xr)</td><td>' + visualize_sample_reconstruction(self.X, self.Xr, list(range(self.X.shape[1])), silent = True) + '</td><tr>'
            html += tr
            tr = '<tr><td>Relative Reconstruction Error (rMSE)</td><td>' + str(round(self.rmse,3)) + '</td><tr>'
            html += tr
        
        tr = '<tr><td>Distance Matrices (D, Dz)</td><td>' + visualize_matrix(self.D, silent = True) + visualize_matrix(self.Dz, silent = True) + '</td><tr>'
        html += tr
        
        tr = '<tr><td>Residual Variance of Distance Matrices</td><td>' + str(round(self.Vr,3)) + ', ' + str(round(self.Vrs,3)) + '</td><tr>'
        html += tr

        tr = '<tr><td>Ranking Matrices (R, Rz)</td><td>' + visualize_matrix(self.R, silent = True) + visualize_matrix(self.Rz, silent = True) + '</td><tr>'
        html += tr
        
        tr = '<tr><td>Co-ranking Matrix (Q)</td><td>' + visualize_matrix(self.Q, silent = True) + '</td><tr>'
        html += tr    

        plt.figure()
        plt.plot(list(range(1, len(self.T) + 1)), self.T, label = 'trustworthiness')
        plt.plot(list(range(1, len(self.C) + 1)), self.C, label = 'continuity')
        plt.ylim(-0.05,1.05)
        plt.legend()
        tr = '<tr><td>Trustworthiness and Continuity</td><td>' + plt2html(plt) + '</td><tr>'
        html += tr
        # plt.show()
        tr = '<tr><td>T AUC</td><td>' + str(round(self.AUC_T,3)) + '</td><tr>'
        html += tr
        tr = '<tr><td>C AUC</td><td>' + str(round(self.AUC_C,3)) + '</td><tr>'
        html += tr
        
        plt.figure()
        plt.ylim(-0.05,1.1)
        plt.plot(list(range(1, len(self.QNN) + 1)), self.QNN)
        plt.ylim(-0.05, 1.05)
        # plt.xlim(1, len(self.QNN) + 1)
        tr = '<tr><td>QNN(k)</td><td>' + plt2html(plt) + '</td><tr>'
        html += tr
        # plt.show()
        
        tr = '<tr><td>QNN AUC</td><td>' + str(round(self.AUC,3)) + '</td><tr>'
        html += tr
        
        plt.figure()
        plt.ylim(-0.05,1.1)
        plt.plot(list(range(1, len(self.LCMC) + 1)), self.LCMC)
        tr = '<tr><td>LCMC(k)</td><td>' + plt2html(plt) + '</td><tr>'
        html += tr
        # plt.show()
        
        tr = '<tr><td>kmax</td><td>' + str(round(self.kmax,3)) + '</td><tr>'
        html += tr

        tr = '<tr><td>Qlocal</td><td>' + str(round(self.Qlocal,3)) + '</td><tr>'
        html += tr

        tr = '<tr><td>Qglobal</td><td>' + str(round(self.Qglobal,3)) + '</td><tr>'
        html += tr


        ##### PCA Explained Variance (used to determine optimal k) ######
        html += "</table><hr/><table>"

        from sklearn.decomposition import PCA
        import matplotlib.ticker as mticker

        K = min(50, self.X.shape[1], self.X.shape[0])
        pca = PCA(n_components = K) # keep the first K components
        pca.fit(self.X)
        X_pca = pca.transform(self.X)

        plt.figure(figsize=(10,4))
        plt.scatter(list(range(1,K+1)), pca.explained_variance_ratio_, alpha=0.7, label='variance percentage')
        plt.scatter(list(range(1,K+1)), pca.explained_variance_ratio_.cumsum(), alpha=0.5, label='cumulated variance percentage')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
        plt.legend()
        plt.title('PCA explained variance against k (number of components). \nYou may use this curve to decide the optimal k.')
        # plt.show()
        # print('explained variance ratio:', pca.explained_variance_ratio_) 
        
        tr = '<tr><td>' + plt2html(plt) + '</td><tr>'
        html += tr

        return html + "</table>"

    def get_json(self):
        dic = {}

        for key in self.__dict__:
            if (key in ['X', 'Z', 'Xr', 'D', 'Dz', 'R', 'Rz', 'Q', 'T', 'C', 'QNN', 'LCMC']):
                continue
            if isinstance(self.__dict__[key], np.ndarray):
                dic[key] = self.__dict__[key].tolist()
            elif isinstance(self.__dict__[key], np.float64) or isinstance(self.__dict__[key], np.int64):
                dic[key] = self.__dict__[key].item()
            else:
                dic[key] = self.__dict__[key]
        
        return json.dumps(dic)