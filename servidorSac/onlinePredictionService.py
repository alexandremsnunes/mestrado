#!/usr/bin/env python
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

# Nome do arquivo que armazenará a coleta do SAC-DM
nomeArquivo = 'finalized_model.sav'

localIP     = "192.168.1.100"

localPort   = 20002

bufferSize  = 1024

# Create a datagram socket

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

class_names = ['Normal','Desbalanceado']
# Bind to address and ip

UDPServerSocket.bind((localIP, localPort))


x=[]
y=[]
z=[]
timestamp = []

print("UDP server up and listening")
# Cria thread que monitora o fechamento do programa com 'Ctrl + C'
# Quando programa for fechado, executa essa função:
def signal_handler(signal, frame):
    #print('\n\nSalvando coleta de dados em {}.csv'.format(nomeArquivo))
    """ for j in range(len(timestamp)):
        df.loc[len(df)] = [str(timestamp[j]), x[j], y[j], z[j]] # Adiciona os novos valores em uma nova linha do DataFrame """
    #df.to_csv('{}.csv'.format(nomeArquivo))
    sys.exit(0)

def sac_dm(data):
	M = len(data)
	N = 1000
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

signal.signal(signal.SIGINT, signal_handler)
forever = threading.Event()

# Cria DataFrame com as colunas referentes aos dados da coleta
df = pd.DataFrame(columns=['Timestamp','Eixo X', 'Eixo Y', 'Eixo Z'])



# Cria rota que a ESP8266 realizará o http.POST com os resultados do SAC-DM
temp_ini = time.time()

classifier = pickle.load(open(nomeArquivo, 'rb'))



input = [[],[],[]]
     
while(True):
    
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
    
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]
    
    tempo_fim = time.time() - temp_ini
    #timestamp.append(time.time() - temp_ini)

    eixos = format(message)
    eixos = eixos.split("'")

    if (len(input[0]) <= 1000):
        #LOW M5 BATTERY
        if(eixos[1] == "LOW_BATTERY" or eixos[1] == "end"):
            if(eixos[1]  == "end"):
                print('Usuario finalizou!')
                #msgFromClient = "ending"
                #print(msgFromClient)
                #bytesToSend = str.encode(msgFromClient)
                #UDPServerSocket.sendto(bytesToSend, address)
                #print("entrei no else")
            else:
                print("|M5| > LOW_BATTERY {}".format(tempo_fim))
            
            #print('\n\nSalvando coleta de dados em {}.csv'.format(nomeArquivo))        
            #df.to_csv('{}.csv'.format(nomeArquivo))
            sys.exit(0)

        else:
            eixos = eixos[1].split(',') 
            #print(eixos)
            #print(eixos)

            if (len(input[0])==0):
                #print("entrei IF")
                input[0].append(eixos[0])
                input[1].append(eixos[1])
                input[2].append(eixos[2])

            else:
                
                if(eixos[0] !=  input[0][len(input[0])-1] or input[1][len(input[1])-1] or eixos[2] != input[2][len(input[2])-1]):
                    #print(len(df))
                    input[0].append(eixos[0])
                    input[1].append(eixos[1])
                    input[2].append(eixos[2])

    else:
        
        input[0],input[1],input[2] = sac_dm(input[0]),sac_dm(input[1]),sac_dm(input[2])
        X = np.array(input).T 
        X = X.astype(np.float64)
        res = classifier.predict(X)
        print("Voo: ",class_names[res[0]])
        input = [[],[],[]]