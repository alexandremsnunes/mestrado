#!/usr/bin/env python
from contextlib import nullcontext
import re
from mqtt import sendMQTT
import numpy as np
import pickle
# Importa pacotes para criação do servidor e manipulação dos dados
import socket
import time
import pandas as pd
import paho.mqtt.client as paho
# Importa pacotes responsáveis por cuidar da interrupção quando o programa for fechado com 'Ctrl + C'
import signal
import sys
import threading
import statistics
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

#from sacdmClassifierAxiesByCSV import X_test

# Nome do arquivo que armazenará a coleta do SAC-DM
nomeArquivo = ['KNN_final.sav','X.sav','teste']

localIP     = "192.168.43.142"

localPort   = 20001

bufferSize  = 1024


# Create a datagram socket

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

class_names = ["Drone Repouso", "Voo Normal", "Voo Falha 1", "Voo Falha 2","Voo Falha 2"]
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
	N = 100
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

def sac_am(data):
    M = len(data)
    N = 100

    size = int(M/N)
    sacam = [0.0] * size

    start = 0
    end = N

    for k in range(size):
    
        peaks, _ = find_peaks(data[start:end])
        v = []
        for p in range(len(data[peaks])): v.append(data[peaks][p]) 
        s = sum(np.absolute(v))
        sacam[k] = 1.0*s/N
        start = end
        end += N

    return sacam

signal.signal(signal.SIGINT, signal_handler)
forever = threading.Event()

# Cria DataFrame com as colunas referentes aos dados da coleta
df = pd.DataFrame(columns=['Timestamp','Eixo X', 'Eixo Y', 'Eixo Z'])



# Cria rota que a ESP8266 realizará o http.POST com os resultados do SAC-DM
temp_ini = time.time()

classifier = pickle.load(open(nomeArquivo[0], 'rb'))
#data_teste = pickle.load(open(nomeArquivo[1], 'rb'))



input = [[],[],[]]
     
while(True):
    
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
    
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]
    
    tempo_fim = time.time() - temp_ini
    #timestamp.append(time.time() - temp_ini)

    eixos = format(message)
    eixos = eixos.split("'")

    if(len(df) <= 3000):
        
        #LOW M5 BATTERY
        if(eixos[1] == "LOW_BATTERY" or eixos[1] == "end"):
            if(eixos[1]  == "end"):
                print('Usuario finalizou!')
                msgFromClient = "ending"
                print(msgFromClient)
                bytesToSend = str.encode(msgFromClient)
                UDPServerSocket.sendto(bytesToSend, address)
                #print("entrei no else")
            else:
                print("|M5| > LOW_BATTERY {}".format(tempo_fim))
        
            print('\n\nSalvando coleta de dados em {}.csv'.format(nomeArquivo))        
            df.to_csv('{}.csv'.format(nomeArquivo))
            sys.exit(0)

        else:
            eixos = eixos[1].split(',') 
            #print(eixos)
            #print(eixos)

            if (len(df)==0):
                print("Recebendo dados...")
                df.loc[len(df)] = [str(tempo_fim), eixos[0], eixos[1], eixos[2]]

            else:
                
                if(eixos[0] != df.loc[len(df)-1, 'Eixo X'] or eixos[1] != df.loc[len(df)-1, 'Eixo Y'] or eixos[2] != df.loc[len(df)-1, 'Eixo Z']):
                    #print(len(df))
                    df.loc[len(df)] = [str(tempo_fim), eixos[0], eixos[1], eixos[2]]

    else:
        df.to_csv('{}.csv'.format(nomeArquivo[2]))
        data1 = pd.read_csv(nomeArquivo[2]+".csv")
        input[0],input[1],input[2] = sac_am(np.array(data1["Eixo X"])),sac_am(np.array(data1["Eixo Y"])),sac_am(np.array(data1["Eixo Z"]))
        X = np.array(input).T 
        X = X.astype(np.float64)
        res = classifier.predict(X)
        sendMQTT(class_names[statistics.mode(res)])
        df = df[0:0]