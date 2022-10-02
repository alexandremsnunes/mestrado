from contextlib import nullcontext
import sys
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

from scipy import signal
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


N = 1000

def sac_am(data):
    M = len(data)
    
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

input1 = [[],[],[]]
input2 = [[],[],[]]
input3 = [[],[],[]]
input4 = [[],[],[]]

file1 = sys.argv[1]+'.csv'
data1 = pd.read_csv(file1)
class_names = [sys.argv[1]]
input1[0],input1[1],input1[2] = sac_dm(np.array(data1["Eixo X"])),sac_dm(np.array(data1["Eixo Y"])),sac_dm(np.array(data1["Eixo Z"]))
if(len(sys.argv) > 2):
    file2 = sys.argv[2]+'.csv'
    data2 = pd.read_csv(file2)
    class_names = [sys.argv[1],sys.argv[2]]    
    input2[0],input2[1],input2[2] = sac_dm(np.array(data2["Eixo X"])),sac_dm(np.array(data2["Eixo Y"])),sac_dm(np.array(data2["Eixo Z"]))

if(len(sys.argv) > 3):
    file3 = sys.argv[3]+'.csv'
    data3 = pd.read_csv(file3)
    class_names = ["VN","CF1","CF2"]
    input3[0],input3[1],input3[2] = sac_dm(np.array(data3["Eixo X"])),sac_dm(np.array(data3["Eixo Y"])),sac_dm(np.array(data3["Eixo Z"]))

if(len(sys.argv) > 4):
    file4 = sys.argv[4]+'.csv'
    data4 = pd.read_csv(file4)
    class_names = [sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]
    input4[0],input4[1],input4[2] = sac_dm(np.array(data4["Eixo X"])),sac_dm(np.array(data4["Eixo Y"])),sac_dm(np.array(data4["Eixo Z"]))



#data3 = pd.read_csv(file3)





#inputChegada = [[],[],[]]


#inputChegada[0],inputChegada[1],inputChegada[2] = sac_dm(np.array(data3["Eixo X"])),sac_dm(np.array(data3["Eixo Y"])),sac_dm(np.array(data3["Eixo Z"]))
aux = [len(input1[0]),len(input2[0]),len(input3[0]),len(input4[0])]
for i in range(aux.count(0)): aux.remove(0) 

#menorTam = min(aux)
menorTam = int(len(input1[0])/2)

if(len(input1[0]) != menorTam):
    input1[0],input1[1],input1[2] = input1[0][:menorTam],input1[1][:menorTam],input1[2][:menorTam]
if(len(input2[0]) != menorTam and len(input2[0])> 0):
    input2[0],input2[1],input2[2] = input2[0][:menorTam],input2[1][:menorTam],input2[2][:menorTam]
if(len(input3[0]) != menorTam and len(input3[0])> 0):
    input3[0],input3[1],input3[2] = input3[0][:menorTam],input3[1][:menorTam],input3[2][:menorTam]
if(len(input4[0]) != menorTam and len(input4[0])> 0):
    input4[0],input4[1],input4[2] = input4[0][:menorTam],input4[1][:menorTam],input4[2][:menorTam]



X = np.concatenate((np.array(input1).T, np.array(input2).T, np.array(input3).T, np.array(input4).T)) 
X = X.astype(np.float64)
print(X)


i1 = [0]*len(input1[0])
i2 = [1]*len(input2[0])
i3 = [2]*len(input3[0])
i4 = [4]*len(input4[0])

y = i1 + i2 + i3 + i4

y = np.array(y)





# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
#classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
#classifier = KNeighborsClassifier(2).fit(X_train, y_train)
#classifier = MLPClassifier(alpha=1, max_iter=1000).fit(X_train, y_train)

# Classificadores 

classifier = KNeighborsClassifier(3).fit(X_train, y_train)
#classifier = svm.SVC(gamma=2, C=1000).fit(X_train, y_train)
#classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

#usar esse


#classifier = GaussianProcessClassifier(1.0 * RBF(1.0)).fit(X_train, y_train)

#classifier = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)


#classifier = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

#classifier = MLPClassifier(alpha=1, max_iter=1000).fit(X_train, y_train)

#classifier = AdaBoostClassifier(n_estimators=100,learning_rate=0.5).fit(X_train, y_train) #esse ta melhor

#classifier = GaussianNB().fit(X_train, y_train)

#classifier = QuadraticDiscriminantAnalysis().fit(X_train, y_train)

#filename = 'finalized_model.sav'
#pickle.dump(classifier, open(filename, 'wb'))

#codigo para ler o modelo gerado

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Matriz de Confusão com SAC-AM", 'true')]



for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    if (title == "Confusion matrix, without normalization"):
        cfm = disp.confusion_matrix


print("acuracia: ",classifier.score(X, y),"%")

filename = 'KNN_final.sav'
pickle.dump(classifier, open(filename, 'wb'))
#print(len(X_test),len(X))
#loaded_model = pickle.load(open("KNN.sav", 'rb'))
#xteste = pickle.load(open("X.sav", 'rb'))
#result = loaded_model.predict(xteste)

#for i in range(len(result)): print(result[i],y_test[i])

#print(result)


""" B = np.array(inputChegada).T 
B = B.astype(np.float64)


res = classifier.predict(B)

for i in range(len(res)):
    #print(res[i])
    print("Voo: ",class_names[res[i]])

print( ((np.count_nonzero(res == 0))/len(res))*100)

 """


plt.show()
 