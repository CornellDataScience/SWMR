from matplotlib import *
import sys
from pylab import *
import numpy as np; np.random.seed(0)
import seaborn as sns
from matplotlib.figure import Figure
import pandas as pd
from sklearn.metrics import confusion_matrix
# in the columnn 14, 15 there's no actual value

def conmatrix(confusion):
    '''
    This function is only for 20 x 20 pre x y_test
    '''
    conmatrix = np.zeros((20,20))
    for j in range(20):
        for i in range(20):
            conmatrix[i,j]=confusion.iloc[i,j]/(sum(confusion.iloc[:,j]))
    conmatrix=pd.DataFrame(conmatrix )
    return conmatrix

def make_heatmap_confusion(y_test, y_pred):
    confusion=pd.DataFrame(confusion_matrix(y_test, y_pred))
    percen = conmatrix(confusion)
    plt.figure(figsize=(20,15))
    plt.title("Probability for Classes")
    sns.set()
    ax = sns.heatmap(percen,annot= True, cmap="BuPu")
