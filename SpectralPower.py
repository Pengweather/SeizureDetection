import gen_filt as gf
import pandas as pd
import numpy as np

def calculateFeatureValue(Feature, cutoffFreqStart, cutoffFreqEnd):
	x = gf.filt_data(Feature.Measurement.seizureData, cutoffFreqStart, cutoffFreqEnd, Feature.Measurement.Fs, 10)
	s = pd.Series(x**2)
	energy = s.rolling(Feature.WindowLength.astype(int) - 1).mean()
	energy = energy[::Feature.StepSize]
	#print(np.asarray(energy).shape)
	return np.asarray(energy)
