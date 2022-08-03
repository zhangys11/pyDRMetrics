# pyDRMetrics

pyDRMetrics - A Python toolkit for dimensionality reduction quality assessment, Heliyon, Volume 7, Issue 2, 2021, e06199, ISSN 2405-8440, https://doi.org/10.1016/j.heliyon.2021.e06199. (https://www.sciencedirect.com/science/article/pii/S2405844021003042)


A more friendly GUI tool using pyDRMetrics can be accessed at http://spacs.brahma.pub/research/DR

File list:

  src/pyDRMetrics.py - the main module  
  src/other py files - dependent modules  
  data/ovarian-cancer-nci-pbsii-data-no-header.csv - SELDI-TOF-MS dataset used in the case study. 253 samples. Each sample has 15154 dimensions.  
  data/cancer.csv - A subset of ovarian-cancer-nci-pbsii-data containing 10 normal and 10 cancer samples. DOI: 10.1016/S0140-6736(02)07746-2  
  data/digits.csv - 40 samples from the MNIST handwritten digits dataset. URL: http://yann.lecun.com/exdb/mnist/  
  data/raman.csv - Another dataset containing the Raman spectra of 46 infant formula milk powder samples. DOI: 10.1016/j.talanta.2019.120681   
  doc.pdf - the code and result for the case study

# Installation

`pip install pyDRMetrics`

# How to use this package (with sample code):

1. Download any sample dataset from the /data folder
2. Use the following sample code to use the package:
<pre>

# import the library
from pyDRMetrics.pyDRMetrics import *

# load the dataset
import pandas as pd
data = pd.read_csv('raman.csv')
cols = data.shape[1]
# convert from pandas dataframe to numpy matrices
X = np.array(data.iloc[:,1:-1]) # skip first and last cols
y = np.array(data.iloc[:,-1])
X_names = list(data.columns.values[1:-1]) # -1 for removing the last column
labels = list(set(y))

# perform DR, e.g., PCA
from sklearn.decomposition import PCA
import matplotlib.ticker as mticker
K = 2
pca = PCA(n_components = K) # keep the first K components
pca.fit(X)
Z = pca.transform(X)
Xr = pca.inverse_transform(Z)

# Create DRMetrics object. This object contains all DR metrics and main API functions
drm = DRMetrics(X, Z, Xr)
drm.report() # this will generate a detailed report. You can also access each metric, e.g., drm.QNN, drm.LCMC, etc.

</pre>

You may also check doc.pdf for more sample codes.
