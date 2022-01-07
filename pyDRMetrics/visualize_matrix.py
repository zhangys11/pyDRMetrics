import matplotlib.pyplot as plt
from .plt2base64 import *

def visualize_matrix(dm, cmap = None, silent = False):

    fig = plt.figure(figsize = (3,3))
    if cmap is None:
        plt.imshow(dm)
    else:
        plt.imshow(dm, cmap = cmap)

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False) 

    fig.tight_layout()
        
    s = plt2html(plt) # must appear before plt.show()

    if (silent == False):
        plt.show()
    return s