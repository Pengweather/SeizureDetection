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
from functools import reduce
from sklearn import svm
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', type=str, default= 'trained')
parser.add_argument('--Start', '-s', type=int, default=11)
parser.add_argument('--End', '-e', type=int, default=15)
args = parser.parse_args()
file = args.filename


NUM_CONFIG = 1
# Uploading the testing data
MeasObjCh1 = mm.Measurement('Study_005_channel1.pkg', args.Start, args.End)
MeasObjCh1.downsample(2)
#print(len(MeasObjCh1.seizureData))
#print(len(MeasObjCh1.label))

# Calculating all the relevant features
FeatObj1 = fe.Feature(MeasObjCh1)
#print(len(FeatObj1.labelDownsampled))

thetaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 4, 8)
alphaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 14, 32)
betaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 8, 12)

nonlinearEnergyFeature1 = ny.calculateFeatureValue(FeatObj1)
lineLengthFeature1 = ll.calculateFeatureValue(FeatObj1, FeatObj1.stepSize.astype(int), FeatObj1.windowLength.astype(int))

# Weed out all the bad values here
thetaBandPowerFeature1IsNaN = np.where(np.isnan(thetaBandPowerFeature1))
alphaBandPowerFeature1IsNaN = np.where(np.isnan(alphaBandPowerFeature1))
betaBandPowerFeature1IsNaN = np.where(np.isnan(betaBandPowerFeature1))

nonlinearEnergyFeature1IsNaN = np.where(np.isnan(nonlinearEnergyFeature1))
lineLengthFeature1IsNaN = np.where(np.isnan(lineLengthFeature1))

indicesToRemove = reduce(np.union1d, (thetaBandPowerFeature1IsNaN[0], alphaBandPowerFeature1IsNaN[0], betaBandPowerFeature1IsNaN[0], nonlinearEnergyFeature1IsNaN[0], lineLengthFeature1IsNaN[0]))

#print(indicesToRemove)
#print(thetaBandPowerFeature1.shape)

# Removing the indices
for i in sorted(indicesToRemove.tolist(), reverse = True):
	thetaBandPowerFeature1 = np.delete(thetaBandPowerFeature1, i)
	alphaBandPowerFeature1 = np.delete(alphaBandPowerFeature1, i)
	betaBandPowerFeature1 = np.delete(betaBandPowerFeature1, i)
	nonlinearEnergyFeature1 = np.delete(nonlinearEnergyFeature1, i)
	lineLengthFeature1 = np.delete(lineLengthFeature1, i)
	FeatObj1.labelDownsampled = np.delete(FeatObj1.labelDownsampled, i)

# Do the feature normalization here
#print(features.shape)
#print(lineLengthFeature1[:5])
#print(features[:5])
#print(FeatObj1.labelDownsampled[:5])

Accu = []
FP = []
tp = []

# Upload the results from training
for i in range(NUM_CONFIG):
	filename = file + "_" + str(i) + ".pkg"
	loadData = pickle.load(open(filename, 'rb'))
	clf = loadData['model']
	Method = loadData['Method']
	Norm = loadData['Norm']
	if Norm == "MeanStd":
		print("Using MeanStd")
		#print(loadData['mean'])
		thetaBandPowerFeature1 = Normalization.normalizeDataMeanStd(np.asarray(thetaBandPowerFeature1),loadData['mean'][0],loadData['std'][0])
		alphaBandPowerFeature1 = Normalization.normalizeDataMeanStd(np.asarray(alphaBandPowerFeature1),loadData['mean'][1],loadData['std'][1])
		betaBandPowerFeature1 = Normalization.normalizeDataMeanStd(np.asarray(betaBandPowerFeature1),loadData['mean'][2],loadData['std'][2])

		nonlinearEnergyFeature1 = Normalization.normalizeDataMeanStd(np.asarray(nonlinearEnergyFeature1),loadData['mean'][3],loadData['std'][3])
		lineLengthFeature1 = Normalization.normalizeDataMeanStd(np.asarray(lineLengthFeature1),loadData['mean'][4],loadData['std'][4])
###############################################################################################################################
	elif Norm == "MinMax":
		print("Using MinMax")
		thetaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(thetaBandPowerFeature1))
		alphaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(alphaBandPowerFeature1))
		betaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(betaBandPowerFeature1))

		nonlinearEnergyFeature1 = Normalization.normalizeDataMinMax(np.asarray(nonlinearEnergyFeature1))
		lineLengthFeature1 = Normalization.normalizeDataMinMax(np.asarray(lineLengthFeature1))
	###############################################################################################################################
	features = np.reshape(np.hstack((thetaBandPowerFeature1,alphaBandPowerFeature1, betaBandPowerFeature1, nonlinearEnergyFeature1,lineLengthFeature1)),(-1,5),1)



	result = clf.predict(features)
	#print(result.shape)
	Accu_temp, FP_temp = FeatObj1.analyze(result)
	Accu.append(Accu_temp)
	FP.append(FP_temp)




if Method == "Regress" :
	Accu = []
	FP = []
	for i in range (10):
		threshold = 0.09 + i * 0.1
		predict = np.asarray(result + (1-threshold)).astype(int)
		Accu_temp, FP_temp = FeatObj1.analyze(predict)
		Accu.append(Accu_temp)
		FP.append(FP_temp)
plt.figure()
plt.xlabel('Index')
plt.ylabel('Label')
plt.title('Actual Label vs Prediction' + "(" + Method + ')')

plt.plot(result,color = 'r',label = 'Predicted')
plt.plot(FeatObj1.labelDownsampled,color = 'g',label = 'Actual')
plt.legend(loc='upper left')
#plt.scatter(tp,np.ones(len(tp)),marker = "*")
plt.show()

if Method == "Regress" :
	print(FP)
	plt.figure()
	plt.title('Sensiticity vs False Alarm')
	plt.xlabel('False Positive Rate')
	plt.ylabel('Sensitivity')
	plt.plot(FP, Accu, marker = "*")
	plt.show()
print("Sensitivity = " + str(Accu_temp*100) + "%")
print("FP = " + str(FP_temp*100) + "%")
