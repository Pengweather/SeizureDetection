import Feature as fe
import Measurement as mm
import pandas as pd
import numpy as np
import scipy as sp
import math

def calculateFeatureValue(Feature):
	data = Feature.Measurement.seizureData
	val = np.square(data) - (np.roll(data, 1) * np.roll(data, -1))
	val[0] = 0
	val[-1] = 0
	# For now, use this method from Pandas
	movAverage = pd.rolling_mean(data, Feature.WindowLength.astype(int) - 1)
	# This results in a memory error
	# movAverage = ma.movingAverage(val, self.WindowLength.astype(int) - 1, 0)
	print(np.array(movAverage).size)
	movAverage = np.array(movAverage)
	value = movAverage[range(0, movAverage.size - 1, Feature.StepSize.astype(int))]
	return value

"""
MeasObj = mm.Measurement('Study_005_channel1.txt')
MeasObj.downsample(2)
FeatObj = fe.Feature(MeasObj)
calculateFeatureValue(FeatObj)
"""
