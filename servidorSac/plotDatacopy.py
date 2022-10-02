
# Standard python numerical analysis imports:
from contextlib import nullcontext

import numpy as np
import pickle
# Importa pacotes para criação do servidor e manipulação dos dados
import socket
import time
import pandas as pd

# Importa pacotes responsáveis por cuidar da interrupção quando o programa for fechado com 'Ctrl + C'
import signal
import sys
import threading

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

#from scipy import signal
import scipy.interpolate #import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
from scipy.signal import find_peaks, peak_prominences
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#from scipy.interpolate import spline

def sac_dm(data):
	M = len(data)
	
	size = int(M/N)
	sacdm=[0.0] * size

	start = 0
	end = N
	for k in range(size):
		peaks, _ = find_peaks(data[start:end])
		v = np.array(peaks)
		sacdm[k] = (1.0*len(v)/N)
		start = end
		end += N

	return sacdm

def sac_am(data, N):
    M = len(data)
    
    size = int(M/N)
    sacam = [0.0] * size

    start = 0
    end = N

    for k in range(size):
    
        peaks, _ = find_peaks(data[start:end])
        v = []
        for p in range(len(data[peaks])): v.append(data[peaks][p][0]) 
        s = sum(np.absolute(v))
        sacam[k] = 1.0*s/N
        start = end
        end += N

    return sacam

	


#********* Main ********

N = 100
classifier = pickle.load(open('KNN.sav', 'rb'))
#N = int(sys.argv[3])

input1 = [[],[],[]]
file1 = sys.argv[1]+'.csv'
data1 = pd.read_csv(file1)
class_names = [sys.argv[1]]
input1[0],input1[1],input1[2] = sac_dm(np.array(data1["Eixo X"])),sac_dm(np.array(data1["Eixo Y"])),sac_dm(np.array(data1["Eixo Z"]))

X = np.array(input1).T 
X = X.astype(np.float64)

res = classifier.predict(X)

print(res)
