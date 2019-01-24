import Feature
import pandas as pd
import numpy as np
import scipy as sp
import math

def calculateFeatureValue(feature):
	data = feature.measurement.seizureData
	val = np.square(data) - (np.roll(data, 1) * np.roll(data, -1))
	val[0] = 0
	val[-1] = 0
	# For now, use this method from Pandas
	s = pd.Series(val) 
	movAverage = s.rolling(feature.windowLength.astype(int) - 1).mean()
	movAverage = movAverage[::feature.stepSize]
	# This results in a memory error
	# movAverage = ma.movingAverage(val, self.WindowLength.astype(int) - 1, 0)
	print(np.asarray(movAverage).shape)
	movAverage = np.array(movAverage)
	# value = movAverage[range(0, movAverage.size - 1, Feature.StepSize.astype(int))]
	return movAverage

"""
MeasObj = mm.Measurement('Study_005_channel1.txt')
MeasObj.downsample(2)
FeatObj = fe.Feature(MeasObj)
calculateFeatureValue(FeatObj)
"""
