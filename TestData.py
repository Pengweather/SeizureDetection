from functools import reduce
from sklearn import svm

import Measurement as mm
import NonlinearEnergy as ny
import SpectralPower as sr
import Feature as fe
import LineLength as ll
import numpy as np
import scipy as sp
import scipy.signal as sp
import Normalization
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', type=str, default= 'trained')
parser.add_argument('--Start', '-s', type=int, default=11)
parser.add_argument('--End', '-e', type=int, default=15)
parser.add_argument('--NUM_CONFIG', '-n', type=int, default=2)
args = parser.parse_args()
file = args.filename
NUM_CONFIG = args.NUM_CONFIG

# Uploading the testing data
MeasObjCh1 = mm.Measurement('Study_005_channel1.pkg', args.Start, args.End)
MeasObjCh1.downsample(2)

# Calculating all the relevant features
FeatObj1 = fe.Feature(MeasObjCh1)
thetaBandPowerFeature1 = sr.calcValue(FeatObj1, 4, 8)
alphaBandPowerFeature1 = sr.calcValue(FeatObj1, 14, 32)
betaBandPowerFeature1 = sr.calcValue(FeatObj1, 8, 12)
nonlinearEnergyFeature1 = ny.calcValue(FeatObj1)
lineLengthFeature1 = ll.calcValue(FeatObj1, FeatObj1.stepSize.astype(int), FeatObj1.windowLength.astype(int))

# Weed out all the bad values here
thetaBandPowerFeature1IsNaN = np.where(np.isnan(thetaBandPowerFeature1))
alphaBandPowerFeature1IsNaN = np.where(np.isnan(alphaBandPowerFeature1))
betaBandPowerFeature1IsNaN = np.where(np.isnan(betaBandPowerFeature1))
nonlinearEnergyFeature1IsNaN = np.where(np.isnan(nonlinearEnergyFeature1))
lineLengthFeature1IsNaN = np.where(np.isnan(lineLengthFeature1))
indicesToRemove = reduce(np.union1d, (thetaBandPowerFeature1IsNaN[0], alphaBandPowerFeature1IsNaN[0], betaBandPowerFeature1IsNaN[0], nonlinearEnergyFeature1IsNaN[0], lineLengthFeature1IsNaN[0]))

data = MeasObjCh1.seizureData[::FeatObj1.stepSize]

# Removing the indices
for i in sorted(indicesToRemove.tolist(), reverse = True):
	thetaBandPowerFeature1 = np.delete(thetaBandPowerFeature1, i)
	alphaBandPowerFeature1 = np.delete(alphaBandPowerFeature1, i)
	betaBandPowerFeature1 = np.delete(betaBandPowerFeature1, i)
	nonlinearEnergyFeature1 = np.delete(nonlinearEnergyFeature1, i)
	lineLengthFeature1 = np.delete(lineLengthFeature1, i)
	FeatObj1.labelDownsampled = np.delete(FeatObj1.labelDownsampled, i)
	data = np.delete(data, i)

# Upload the results from training
kernels = ['rbf']
for i in range(NUM_CONFIG):
	if not i == 0:
		plt.show(block=False)
	filename = file + "_" + kernels[i] + "_gamma_DEFAULT" + ".pkg"
	loadData = pickle.load(open(filename, 'rb'))
	clf = loadData['model']
	Method = loadData['method']
	Norm = loadData['norm']
	if Norm == "MeanStd":
		print("Using MeanStd")
		temp_thetaBandPowerFeature1 = Normalization.normMeanStd(np.asarray(thetaBandPowerFeature1),loadData['mean']['tbp'],loadData['std']['tbp'])
		temp_alphaBandPowerFeature1 = Normalization.normMeanStd(np.asarray(alphaBandPowerFeature1),loadData['mean']['abp'],loadData['std']['abp'])
		temp_betaBandPowerFeature1 = Normalization.normMeanStd(np.asarray(betaBandPowerFeature1),loadData['mean']['bbp'],loadData['std']['bbp'])
		temp_nonlinearEnergyFeature1 = Normalization.normMeanStd(np.asarray(nonlinearEnergyFeature1),loadData['mean']['nonlin'],loadData['std']['nonlin'])
		temp_lineLengthFeature1 = Normalization.normMeanStd(np.asarray(lineLengthFeature1),loadData['mean']['line'],loadData['std']['line'])
	elif Norm == "MinMax":
		print("Using MinMax")
		temp_thetaBandPowerFeature1 = Normalization.normMinMax(np.asarray(thetaBandPowerFeature1))
		temp_alphaBandPowerFeature1 = Normalization.normMinMax(np.asarray(alphaBandPowerFeature1))
		temp_betaBandPowerFeature1 = Normalization.normMinMax(np.asarray(betaBandPowerFeature1))
		temp_nonlinearEnergyFeature1 = Normalization.normMinMax(np.asarray(nonlinearEnergyFeature1))
		temp_lineLengthFeature1 = Normalization.normMinMax(np.asarray(lineLengthFeature1))

	features = np.reshape(np.hstack((temp_thetaBandPowerFeature1, temp_alphaBandPowerFeature1, temp_betaBandPowerFeature1, temp_nonlinearEnergyFeature1, temp_lineLengthFeature1)),(-1,5),1)
	result = clf.predict(features)
	print(np.count_nonzero(result))
	plt.figure()
	plt.xlabel('Index')
	plt.ylabel('Label')
	plt.title('Actual Label vs Prediction' + "(" + Method + ')_' + str(i))
	plt.plot(data)
	plt.plot(np.multiply(data,result), color = 'r', label = 'Predicted')
	plt.plot(FeatObj1.labelDownsampled * max(data) ,color = 'g',label = 'Actual')
	plt.legend(loc='upper left')

# Testing different Threshold for linear regression
'''
if Method == "Lin_Regress" :
	Accu = []
	FP = []
	for i in range (20):
		threshold = 0.09 + i * 0.05
		predict = np.asarray(result + (1-threshold)).astype(int)
		Accu_temp, FP_temp = FeatObj1.analyze(predict)
		Accu.append(Accu_temp)
		FP.append(FP_temp)


print("Sensitivity = " + str(Accu_temp*100) + "%")
print("FP = " + str(FP_temp*100) + "%")
'''
# Plot out Sensitivity and False alarm for different threshold
'''
plt.show(block=False)
plt.figure()
plt.title('Sensiticity vs False Alarm')
plt.xlabel('False Positive Rate')
plt.ylabel('Sensitivity')
plt.plot(FP, Accu, marker = "*")
'''
plt.show()
# MinMax for trained data
# Look at sensitivity and false detection
# Cross validation
