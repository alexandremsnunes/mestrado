
# Standard python numerical analysis imports:
import numpy as np
import pandas as pd

from scipy import signal
import scipy.interpolate #import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
from scipy.signal import find_peaks, peak_prominences
#import pandas as pd
#import peakutils

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import h5py
import statistics

import sys

#from scipy.interpolate import spline

def sac_dm(data, N):
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

#N = int(sys.argv[3])

file1 = sys.argv[1]+'.csv'
file2 = sys.argv[2]+'.csv'
file3 = sys.argv[3]+'.csv'
file4 = sys.argv[4]+'.csv'

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)
data4 = pd.read_csv(file4)

#signal1 = np.array(data1["Eixo X"])
#signal2 = np.array(data2["Eixo X"])
#signal3 = np.array(data2["Eixo X"])


#data1 = np.genfromtxt(file, delimiter=',', names=['t', 'x', 'y','z'])
#data2 = np.genfromtxt(file2, delimiter=',', names=['t', 'x', 'y','z'])
""" if (len(signal1) < len(signal2) ):
	menorTam = len(signal1)
else:
	menorTam = len(signal2)

#menorTam = len(data1)
sac = sac_dm(signal1, N, menorTam)

#menorTam = len(data2)
sac2 = sac_dm(signal2, N, menorTam) """



#Figure Settings
fig, ax = plt.subplots(3,1, sharex=True, sharey=False)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=1.0)
fig.set_size_inches(10, 6, forward=True)
fig.suptitle("Signals" , fontsize=12)


ax[0].plot(data1["Eixo X"],color='r', label= 'Signal x')
ax[0].plot(data2["Eixo X"],color='b', label='Signal x')
ax[0].plot(data3["Eixo X"],color='g', label= 'Signal x')
ax[0].plot(data4["Eixo X"],color='black', label='Signal x')

#ax[0].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax[0].set_title('Signal x')

ax[1].plot(data1["Eixo Y"],color='r', label='Signal y')
ax[1].plot(data2["Eixo Y"],color='b', label='Signal y')
ax[1].plot(data3["Eixo Y"],color='g', label= 'Signal x')
ax[1].plot(data4["Eixo Y"],color='black', label='Signal x')

#ax[1].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax[1].set_title('Signal y')

ax[2].plot(data1["Eixo Z"],color='r', label='Signal z')
ax[2].plot(data2["Eixo Z"],color='b', label='Signal z')
ax[2].plot(data3["Eixo Z"],color='g', label= 'Signal x')
ax[2].plot(data4["Eixo Z"],color='black', label='Signal x')

#ax[2].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax[2].set_title('Signal z')


for ax in ax.flat:
    ax.set(xlabel='n-Value', ylabel='Amplitude')


fig2, ax2 = plt.subplots(3,1, sharex=True, sharey=False)
fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=1.0)
fig2.set_size_inches(10, 6, forward=True)
fig2.suptitle("Signal" , fontsize=12)


ax2[0].plot(sac_dm(data1["Eixo X"],N),color='r', label= 'Signal x')
ax2[0].plot(sac_dm(data2["Eixo X"],N),color='b', label='Signal x')
ax2[0].plot(sac_dm(data3["Eixo X"],N),color='g', label= 'Signal x')
ax2[0].plot(sac_dm(data4["Eixo X"],N),color='black', label='Signal x')
#ax2[0].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax2[0].set_title('Signal x')

ax2[1].plot(sac_dm(data1["Eixo Y"],N),color='r', label='Signal y')
ax2[1].plot(sac_dm(data2["Eixo Y"],N),color='b', label='Signal y')
ax2[1].plot(sac_dm(data3["Eixo Y"],N),color='g', label= 'Signal x')
ax2[1].plot(sac_dm(data4["Eixo Y"],N),color='black', label='Signal x')
#ax2[1].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax2[1].set_title('Signal y')

ax2[2].plot(sac_dm(data1["Eixo Z"],N),color='r', label='Signal z')
ax2[2].plot(sac_dm(data2["Eixo Z"],N),color='b', label='Signal z')
ax2[2].plot(sac_dm(data3["Eixo Z"],N),color='g', label= 'Signal x')
ax2[2].plot(sac_dm(data4["Eixo Z"],N),color='black', label='Signal x')
#ax2[2].legend([sys.argv[1], sys.argv[2]], loc='upper left') 
ax2[2].set_title('Signal z')


for ax2 in ax2.flat:
    ax2.set(xlabel='n-Value', ylabel='Amplitude')


#print("\nFile analyse: " , f , '\n')

plt.show()


""" fig2 = plt.figure()
ax2 = fig2.add_subplot(111)


kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40, ec="k")

plt.hist(sac, **kwargs)
plt.hist(sac2, **kwargs) """

#plt.savefig('alignment.png', format='png')
#plt.show()





	





