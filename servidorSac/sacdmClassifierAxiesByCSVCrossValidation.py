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
from sklearn.model_selection import cross_val_score

N = 1000

def sac_am(data, N, menorTam):
    M = menorTam
    
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



file1 = sys.argv[1]+'.csv'
file2 = sys.argv[2]+'.csv'
file3 = sys.argv[3]+'.csv'
#file4 = sys.argv[4]+'.csv'

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)
#data4 = pd.read_csv(file4)
#data3 = pd.read_csv(file3)

input1 = [[],[],[]]
input2 = [[],[],[]]
input3 = [[],[],[]]
#input4 = [[],[],[]]
#inputChegada = [[],[],[]]

input1[0],input1[1],input1[2] = sac_dm(np.array(data1["Eixo X"])),sac_dm(np.array(data1["Eixo Y"])),sac_dm(np.array(data1["Eixo Z"]))
input2[0],input2[1],input2[2] = sac_dm(np.array(data2["Eixo X"])),sac_dm(np.array(data2["Eixo Y"])),sac_dm(np.array(data2["Eixo Z"]))
input3[0],input3[1],input3[2] = sac_dm(np.array(data3["Eixo X"])),sac_dm(np.array(data3["Eixo Y"])),sac_dm(np.array(data3["Eixo Z"]))
#input4[0],input4[1],input4[2] = sac_dm(np.array(data4["Eixo X"])),sac_dm(np.array(data4["Eixo Y"])),sac_dm(np.array(data4["Eixo Z"]))
#inputChegada[0],inputChegada[1],inputChegada[2] = sac_dm(np.array(data3["Eixo X"])),sac_dm(np.array(data3["Eixo Y"])),sac_dm(np.array(data3["Eixo Z"]))

menorTam = min(len(input1[0]),len(input2[0]),len(input3[0]))
#menorTam = min(len(input1[0]),len(input2[0]),len(input3[0]),len(input4[0]))


if(len(input1[0]) != menorTam):
    input1[0],input1[1],input1[2] = input1[0][:menorTam],input1[1][:menorTam],input1[2][:menorTam]
if(len(input2[0]) != menorTam):
    input2[0],input2[1],input2[2] = input2[0][:menorTam],input2[1][:menorTam],input2[2][:menorTam]
if(len(input3[0]) != menorTam):
    input3[0],input3[1],input3[2] = input3[0][:menorTam],input3[1][:menorTam],input3[2][:menorTam]
#if(len(input4[0]) != menorTam):
#    input4[0],input4[1],input4[2] = input4[0][:menorTam],input4[1][:menorTam],input4[2][:menorTam]



#X = np.concatenate((np.array(input1).T, np.array(input2).T, np.array(input3).T, np.array(input4).T))
X = np.concatenate((np.array(input1).T, np.array(input2).T, np.array(input3).T))
X = X.astype(np.float64)

y = []

for i in range(len(X)):
    if(i<(len(X)/3)):
        y.append(0)    
    elif(i>=(len(X)/3) and i < ((2*len(X))/3)):
        y.append(1)
    else: #elif(i>=((2*len(X))/4) and i < ((3*len(X))/4)):
        y.append(2)
    #else:
    #    y.append(3)

print(y.count(0))
print(y.count(1))
print(y.count(2))
#print(y.count(3))
y = np.array(y)
class_names = [sys.argv[1],sys.argv[2],sys.argv[3]]
# class_names = [sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]




# Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
#classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
#classifier = KNeighborsClassifier(2).fit(X_train, y_train)
#classifier = MLPClassifier(alpha=1, max_iter=1000).fit(X_train, y_train)

# Classificadores 

#classifier = KNeighborsClassifier(3).fit(X_train, y_train)
#classifier = svm.SVC(kernel='linear', C=100).fit(X_train, y_train)

#classifier = svm.SVC(gamma=2, C=1000).fit(X_train, y_train)

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

#np.set_printoptions(precision=2)

#clf = svm.SVC(kernel='linear', C=100, random_state=42)
clf = KNeighborsClassifier(3)
#clf = svm.SVC(kernel='linear', C=100)
scores = cross_val_score(clf, X, y, cv=5)

print(scores)
print(f'{(scores.mean()):.2f}')

# Plot non-normalized confusion matrix
""" titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]



for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    if (title == "Confusion matrix, without normalization"):
        cfm = disp.confusion_matrix """


""" loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result) """



""" B = np.array(inputChegada).T 
B = B.astype(np.float64)


res = classifier.predict(B)

for i in range(len(res)):
    #print(res[i])
    print("Voo: ",class_names[res[i]])

print( ((np.count_nonzero(res == 0))/len(res))*100)
 """

#Figure Settings
""" fig, ax = plt.subplots(3,1, sharex=True, sharey=False)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=1.0)
fig.set_size_inches(10, 6, forward=True)
fig.suptitle("SACDM" , fontsize=12)


ax[0].plot(input1[0],color='r', label= 'Signal x')
ax[0].plot(input2[0],color='b', label='Signal x')
ax[0].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax[0].set_title('Signal x')

ax[1].plot(input1[1],color='r', label='Signal y')
ax[1].plot(input2[1],color='b', label='Signal y')
ax[1].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax[1].set_title('Signal y')

ax[2].plot(input1[2],color='r', label='Signal z')
ax[2].plot(input2[2],color='b', label='Signal z')
ax[2].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax[2].set_title('Signal z')


for ax in ax.flat:
    ax.set(xlabel='n-Value', ylabel='Amplitude')

fig2, ax2 = plt.subplots(3,1, sharex=True, sharey=False)
fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=1.0)
fig2.set_size_inches(10, 6, forward=True)
fig2.suptitle("Signal" , fontsize=12)


ax2[0].plot(data1["Eixo X"],color='r', label= 'Signal x')
ax2[0].plot(data2["Eixo X"],color='b', label='Signal x')
#ax2[0].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax2[0].set_title('Signal x')

ax2[1].plot(data1["Eixo Y"],color='r', label='Signal y')
ax2[1].plot(data2["Eixo Y"],color='b', label='Signal y')
#ax2[1].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax2[1].set_title('Signal y')

ax2[2].plot(data1["Eixo Z"],color='r', label='Signal z')
ax2[2].plot(data2["Eixo Z"],color='b', label='Signal z')
#ax2[2].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax2[2].set_title('Signal z')


for ax2 in ax2.flat:
    ax2.set(xlabel='n-Value', ylabel='Amplitude')


#print("\nFile analyse: " , f , '\n')

plt.show()
 """

 