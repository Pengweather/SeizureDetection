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

MinMax = 1

MeasObjCh1 = mm.Measurement('Study_005_channel1.pkg', 1, 10)
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

# temp means and std
tempMean = [np.mean(thetaBandPowerFeature1),np.mean(alphaBandPowerFeature1),np.mean(betaBandPowerFeature1),np.mean(nonlinearEnergyFeature1),np.mean(lineLengthFeature1)]
tempStd = [np.std(thetaBandPowerFeature1),np.std(alphaBandPowerFeature1),np.std(betaBandPowerFeature1),np.std(nonlinearEnergyFeature1),np.std(lineLengthFeature1)]

# Do the feature normalization here 
if MinMax:
	print("Using MinMax")
	thetaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(thetaBandPowerFeature1))
	alphaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(alphaBandPowerFeature1))
	betaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(betaBandPowerFeature1))

	nonlinearEnergyFeature1 = Normalization.normalizeDataMinMax(np.asarray(nonlinearEnergyFeature1))
	lineLengthFeature1 = Normalization.normalizeDataMinMax(np.asarray(lineLengthFeature1))
else: 
	print("Using MEanStd")
	thetaBandPowerFeature1 = Normalization.normalizeDataMeanStd(thetaBandPowerFeature1, np.mean(thetaBandPowerFeature1), np.std(thetaBandPowerFeature1))
	alphaBandPowerFeature1 = Normalization.normalizeDataMeanStd(alphaBandPowerFeature1, np.mean(alphaBandPowerFeature1), np.std(alphaBandPowerFeature1))
	betaBandPowerFeature1 = Normalization.normalizeDataMeanStd(betaBandPowerFeature1, np.mean(betaBandPowerFeature1), np.std(betaBandPowerFeature1))

	nonlinearEnergyFeature1 = Normalization.normalizeDataMeanStd(nonlinearEnergyFeature1, np.mean(nonlinearEnergyFeature1), np.std(nonlinearEnergyFeature1))
	lineLengthFeature1 = Normalization.normalizeDataMeanStd(lineLengthFeature1, np.mean(lineLengthFeature1), np.std(lineLengthFeature1))
###############################################################################################################################

features = np.reshape(np.hstack((thetaBandPowerFeature1,alphaBandPowerFeature1, betaBandPowerFeature1, nonlinearEnergyFeature1,lineLengthFeature1)),(-1,5),1)
#(thetaBandPowerFeature1))
#print(features.shape)
# print(lineLengthFeature1[:5])
# print(features[:5])
# print(FeatObj1.labelDownsampled[:5])

# This part can be modified for different machine learning architectures
clf = svm.SVC(gamma = 'scale', kernel = 'rbf')
# n_estimators = 10
# clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
y = clf.fit(features, FeatObj1.labelDownsampled)
print("Saving...")
filename = "SVC_trained_0.pkg"
saveData = {'model' : clf, 'mean': tempMean, 'std': tempStd}
pickle.dump(saveData, open(filename, 'wb'))

# False Alarm Rate
# Calculate the sensitivity which is the true positive rate and the specificaty is true negative rate
# Address the matching of the trained and actual data
# Better to calculate those two than accuracy
# Consider them separately
# For the training, it could be possible to "ignore" some of the zeros in the dataset
# Explore the different kernals specifically radial basis function kernal
# Making changes on the shorter datasets 
# Feature normalization
