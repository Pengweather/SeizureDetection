import numpy as np
import scipy as sp
import sys

####################################################################################
#Calculate line length of a given asarray
#input:
#   data:   the data for calcuating line length
#           numpy array with rank 1, length N
#   window: the size of the window for the line length
#           non-zero positive integer
#output:
#   result: the calcuated line length of the given data
#           numpy array with rank 1 and length N-1
#           normalized by window length
#note:
#   The results of |x[i] - x[i-1]| is zero padded in the front with window - 1 zeros
####################################################################################
def calculateFeatureValue(Feature, stepSize = 0.2, window = 1):
	data = Feature.measurement.seizureData
	assert(len(data) > window)
	diff = np.zeros(window-1)
	diff = np.append(diff,np.array(abs(data[1:len(data)] - data[0:len(data)-1])))
	result = [0]
	for i in range(window-1,len(diff)):
		result.append(np.sum(diff[i-window+1:i+1]))
	result = result[::stepSize]
	#print((np.asarray(result)/window).shape)
	return np.asarray(result)/window


####################################################################################
#Performs down sampling on the given data
#input:
#   data:   the data for calcuating line length
#           numpy array with rank 1, length N
#   factor: down sampling rate
#           non-zero positive integer
#output:
#   result: down sampled data
#           numpy array with rank 1 and length N/factor
####################################################################################
def downsample(data, factor = 2):
    return data[::factor]
