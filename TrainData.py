import Measurement as mm
import Feature as fe
import Generation as g

from functools import reduce
from sklearn import svm

import argparse
import numpy as np
import pickle
import time

# An array of keys that will identify each feature
feat_key = ['tbp', 'abp', 'bbp', 'nonlin', 'line']


def train(feat_array, label_downsampled, kernel, gamma = 0):
	# This part can be modified for different machine learning architectures
	if (gamma == 0):
		clf = svm.SVC(gamma = 'scale', kernel = kernel)
		clf.fit(feat_array, label_downsampled)
	else:
		clf = svm.SVC(gamma = gamma, kernel = kernel)
		clf.fit(feat_array, label_downsampled)
	print("Training has been completed")
	return clf

def saveSVM(clf, tempMean, tempStd, norm, kernel, gamma = 0):
	print("Saving...")
	if (gamma == 0):
		filename = "trained_SVM" + kernel + "_gamma_DEFAULT.pkg"
	else:
		filename = "trained_SVM_" + kernel + "_gamma_" + str(gamma).replace('.', '_') + ".pkg"
	saveData = {'model': clf, 'mean': tempMean, 'std': tempStd, 'method': 'SVM', 'norm': norm, 'gamma': gamma}
	pickle.dump(saveData, open(filename, 'wb'))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--Normalize', '-n', type = str, default = 'MeanStd')
	parser.add_argument('--Kernel', '-k', type = str, default = 'rbf')
	parser.add_argument('--Start', '-s', type = int, default = 1)
	parser.add_argument('--End', '-e', type = int, default = 5)

	# This flag determines whether or not multiple trainings will be performed for the
	# cross validation
	parser.add_argument('--CrossValid', '-c', type = bool, default = False)
	parser.add_argument('--GammaMin', type = float, default = 0.0)
	parser.add_argument('--GammaMax', type = float, default = 10)
	parser.add_argument('--Increment', type = float, default = 1.0)

	args = parser.parse_args()

	# Uploading data from a measurement text file
	meas_obj = mm.Measurement("Study_005_channel1.pkg", args.Start, args.End)
	meas_obj.downsample(2)

	# Calculating all the relevant features using information encompassing all the other
	# features
	feat_obj = fe.Feature(meas_obj)

	feat_label_dict = g.getFeaturesAndLabel(meas_obj, feat_obj)
	feat_dict = dict((k, feat_label_dict[k]) for k in feat_key if k in feat_label_dict)
	label_downsampled = feat_label_dict['label']

	# Good practice to check that the correct keys are generated for their value
	if (g.checkDictForFeat(feat_dict) == False):
		assert(False)

	# Storing the temporary means and standard deviations of the features before it is normalized
	# since a dictionary is mutable
	'''
	tempMean = np.asarray([])
	tempStd = np.asarray([])
	for i in feat_key:
			tempMean = np.append(tempMean, np.mean(feat_dict[i]))
			tempStd = np.append(tempStd, np.std(feat_dict[i]))
	'''
	tempMean = {}
	tempStd = {}
	for i in feat_key:
			tempMean[i] = np.mean(feat_dict[i])
			tempStd[i] = np.std(feat_dict[i])

	g.normFeature(feat_dict, args.Normalize, tempMean, tempStd)
	feat_array = g.convertDictToFeatArray(feat_dict)

	if (args.CrossValid == True):
		for i in np.arange(args.GammaMin, args.GammaMax, args.Increment):
			print("Training set for gamma = " + str(i))
			start = time.time()
			c = train(feat_array, label_downsampled, args.Kernel, i)
			end = time.time()
			print(str(end - start) + " seconds")
			saveSVM(c, tempMean, tempStd, args.Normalize, args.Kernel, i)
	else:
		print("Using default Gamma value")
		c = train(feat_array, label_downsampled, args.Kernel)
		saveSVM(c, tempMean, tempStd, args.Normalize, args.Kernel)

if __name__ == "__main__":
	main()
