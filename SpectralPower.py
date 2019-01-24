import GenFilt as gf
import pandas as pd
import numpy as np

def calculateFeatureValue(feature, cutoffFreqStart, cutoffFreqEnd):
	x = gf.filt_data(feature.measurement.seizureData, cutoffFreqStart, cutoffFreqEnd, feature.measurement.Fs, 10)
	s = pd.Series(x**2)
	energy = s.rolling(feature.windowLength.astype(int) - 1).mean()
	energy = energy[::feature.stepSize]
	#print(np.asarray(energy).shape)
	return np.asarray(energy)
