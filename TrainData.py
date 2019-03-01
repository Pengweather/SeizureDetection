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
import sklearn.linear_model as lm
import argparse

def directTrainingSVM(args):
	# Uploading data from a measurement text file
	MeasObjCh1 = mm.Measurement("Study_005_channel1.pkg", args.Start, args.End)
	MeasObjCh1.downsample(2)

	# Calculating all the relevant features using information encompassing all the other
	# features
	FeatObj1 = fe.Feature(MeasObjCh1)

	# Calculating the power within certain frequency bands from the data that are of
	# importance in seizure detection
	thetaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 4, 8)
	alphaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 14, 32)
	betaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 8, 12)

	# Calculating the overall nonlinear energy of the data
	nonlinearEnergyFeature1 = ny.calculateFeatureValue(FeatObj1)

	# Calculating the line length of the data
	lineLengthFeature1 = ll.calculateFeatureValue(FeatObj1, FeatObj1.stepSize.astype(int), FeatObj1.windowLength.astype(int))

	# Weed out all the bad values here
	thetaBandPowerFeature1IsNaN = np.where(np.isnan(thetaBandPowerFeature1))
	alphaBandPowerFeature1IsNaN = np.where(np.isnan(alphaBandPowerFeature1))
	betaBandPowerFeature1IsNaN = np.where(np.isnan(betaBandPowerFeature1))

	nonlinearEnergyFeature1IsNaN = np.where(np.isnan(nonlinearEnergyFeature1))
	lineLengthFeature1IsNaN = np.where(np.isnan(lineLengthFeature1))

	indicesToRemove = reduce(np.union1d, (thetaBandPowerFeature1IsNaN[0], alphaBandPowerFeature1IsNaN[0], betaBandPowerFeature1IsNaN[0], nonlinearEnergyFeature1IsNaN[0], lineLengthFeature1IsNaN[0]))

	# Removing the indices
	for i in sorted(indicesToRemove.tolist(), reverse = True):
		thetaBandPowerFeature1 = np.delete(thetaBandPowerFeature1, i)
		alphaBandPowerFeature1 = np.delete(alphaBandPowerFeature1, i)
		betaBandPowerFeature1 = np.delete(betaBandPowerFeature1, i)
		nonlinearEnergyFeature1 = np.delete(nonlinearEnergyFeature1, i)
		lineLengthFeature1 = np.delete(lineLengthFeature1, i)
		FeatObj1.labelDownsampled = np.delete(FeatObj1.labelDownsampled, i)

	# Storing the temporary means and standard deviations of the features
	tempMean = [np.mean(thetaBandPowerFeature1),np.mean(alphaBandPowerFeature1),np.mean(betaBandPowerFeature1),np.mean(nonlinearEnergyFeature1),np.mean(lineLengthFeature1)]
	tempStd = [np.std(thetaBandPowerFeature1),np.std(alphaBandPowerFeature1),np.std(betaBandPowerFeature1),np.std(nonlinearEnergyFeature1),np.std(lineLengthFeature1)]

	# Do the feature normalization here
	if args.Normalize== "MinMax":
		print("Using MinMax")
		thetaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(thetaBandPowerFeature1))
		alphaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(alphaBandPowerFeature1))
		betaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(betaBandPowerFeature1))
		nonlinearEnergyFeature1 = Normalization.normalizeDataMinMax(np.asarray(nonlinearEnergyFeature1))
		lineLengthFeature1 = Normalization.normalizeDataMinMax(np.asarray(lineLengthFeature1))
	elif args.Normalize == "MeanStd":
		print("Using MeanStd")
		thetaBandPowerFeature1 = Normalization.normalizeDataMeanStd(thetaBandPowerFeature1, np.mean(thetaBandPowerFeature1), np.std(thetaBandPowerFeature1))
		alphaBandPowerFeature1 = Normalization.normalizeDataMeanStd(alphaBandPowerFeature1, np.mean(alphaBandPowerFeature1), np.std(alphaBandPowerFeature1))
		betaBandPowerFeature1 = Normalization.normalizeDataMeanStd(betaBandPowerFeature1, np.mean(betaBandPowerFeature1), np.std(betaBandPowerFeature1))
		nonlinearEnergyFeature1 = Normalization.normalizeDataMeanStd(nonlinearEnergyFeature1, np.mean(nonlinearEnergyFeature1), np.std(nonlinearEnergyFeature1))
		lineLengthFeature1 = Normalization.normalizeDataMeanStd(lineLengthFeature1, np.mean(lineLengthFeature1), np.std(lineLengthFeature1))
	else:
		print("No proper normalization tool was selected")
		assert(False)

	# Reshaping the features array in order to allow classification and training to be possible
	features = np.reshape(np.hstack((thetaBandPowerFeature1,alphaBandPowerFeature1, betaBandPowerFeature1, nonlinearEnergyFeature1,lineLengthFeature1)),(-1,5),1)

	# This part can be modified for different machine learning architecturesor 
	kernels = ['rbf', 'linear']
	# kernels = ['rbf', 'linear', 'poly']
	print('Using SVM')
	for i in range(len(kernels)):
		clf = svm.SVC(gamma = 0.5, kernel = kernels[i])
		clf.fit(features, FeatObj1.labelDownsampled)
		print("Saving...")
		filename ="trained_" + kernels[i] + ".pkg"
		saveData = {'model' : clf, 'mean': tempMean, 'std': tempStd, 'Method': 'SVM', 'Norm': args.Normalize}
		pickle.dump(saveData, open(filename, 'wb'))

parser = argparse.ArgumentParser()
parser.add_argument('--Normalize', '-n', type=str, default= 'MeanStd')
parser.add_argument('--Start', '-s', type=int, default=1)
parser.add_argument('--End', '-e', type=int, default=10) 
parser.add_argument('--CrossValid', '-c', type=bool, default=False)

args = parser.parse_args()
directTrainingSVM(args)
