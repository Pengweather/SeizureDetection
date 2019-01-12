import numpy as np
from joblib import Parallel, delayed

def movingAverage(data, kb = 0, kf = 0):
	dataLength = np.array(data).size
	M = []
        
	for i in range(0, dataLength):
		temp = 0.0
		flagKb = False
		flagKf = False
		kbTemp = kb
		kfTemp = kf
		if (i < kb):
			flagKb = True
			for j in range(i):
				temp = temp + data[j]
			kbTemp = i
		if (i + kf > dataLength - 1):
			flagKf = True
			for j in range(i + 1, dataLength):
				temp = temp + data[j]
			kfTemp = dataLength - i - 1
		if (flagKb == False):
			for j in range(i - kb, i):
				temp = temp + data[j]
		if (flagKf == False):
			for j in range(i + 1, i + kf + 1):
				temp = temp + data[j]
		temp = temp + data[i]
		M.append(temp / (kfTemp + kbTemp + 1))
	
        M = np.array(M)
	return M
