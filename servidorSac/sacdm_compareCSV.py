
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

def sac_dm(data, N, menorTam):
	M = menorTam
	
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



N = int(sys.argv[3])

file1 = sys.argv[1]+'.csv'
file2 = sys.argv[2]+'.csv'


data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

signal1 = np.array(data1["Eixo X"])
signal2 = np.array(data2["Eixo X"])


#data1 = np.genfromtxt(file, delimiter=',', names=['t', 'x', 'y','z'])
#data2 = np.genfromtxt(file2, delimiter=',', names=['t', 'x', 'y','z'])
if (len(signal1) < len(signal2) ):
	menorTam = len(signal1)
else:
	menorTam = len(signal2)

#menorTam = len(data1)
sac = sac_dm(signal1, N, menorTam)

#menorTam = len(data2)
sac2 = sac_dm(signal2, N, menorTam)

print("###### Dados do SAC 1 ######")
print("Media: ", statistics.mean(sac), "Desvio: ", statistics.pstdev(sac))
print("Media + Desvio: ", (statistics.mean(sac)+statistics.pstdev(sac)))
print("Media - Desvio: ", (statistics.mean(sac)-statistics.pstdev(sac)))
print("#############################")
print()
print("###### Dados do SAC 2 ######")
print("Media: ", statistics.mean(sac2), "Desvio: ", statistics.pstdev(sac2))
print("Media + Desvio: ", (statistics.mean(sac2)+statistics.pstdev(sac2)))
print("Media - Desvio: ", (statistics.mean(sac2)-statistics.pstdev(sac2)))
print("#############################")

fig = plt.figure()


ax = fig.add_subplot(211)
ax.set_title(sys.argv[1]+" versus "+  sys.argv[2])   
ax.plot(signal1,color='b', label=sys.argv[1])
ax.plot(signal2,color='r', label=sys.argv[2])
plt.ylabel('Eixo y') 
#plt.xlabel('Time (sec.)')
ax.legend([sys.argv[1], sys.argv[2]], loc='upper right') 
x = np.array(range(0, len(sac)))
x = x * N

x2 = np.array(range(0, len(sac2)))
x2 = x2 * N


ax3 = fig.add_subplot(212)
ax3.plot(x, sac, color='b', label='SAC-DM '+sys.argv[1])
ax3.plot(x2, sac2, color='r', label='SAC-DM '+sys.argv[2])
plt.ylabel('Frequency') 
plt.xlabel('Time (sec.)')
ax3.legend(['SAC-DM '+sys.argv[1], 'SAC-DM '+sys.argv[2]], loc='upper right')



""" fig2 = plt.figure()
ax2 = fig2.add_subplot(111)


kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40, ec="k")

plt.hist(sac, **kwargs)
plt.hist(sac2, **kwargs) """

#plt.savefig('alignment.png', format='png')
plt.show()





	





