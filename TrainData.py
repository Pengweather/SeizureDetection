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
import sklearn.linear_model as lm
import argparse
import matplotlib.pyplot as plt
################################### HELPER FUNCTIONS ###################################
def gen_feature(Norm,Start, End, mean=None,std=None):
	MeasObjCh1 = mm.Measurement("Study_005_channel1.pkg", Start, End)
	MeasObjCh1.downsample(2)
# Calculating all the relevant features
	print('Generating Features ...')
	FeatObj1 = fe.Feature(MeasObjCh1)
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
	print('Normalizing ...')
	if Norm == "MinMax":
		print("Using MinMax")
		thetaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(thetaBandPowerFeature1))
		alphaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(alphaBandPowerFeature1))
		betaBandPowerFeature1 = Normalization.normalizeDataMinMax(np.asarray(betaBandPowerFeature1))
		nonlinearEnergyFeature1 = Normalization.normalizeDataMinMax(np.asarray(nonlinearEnergyFeature1))
		lineLengthFeature1 = Normalization.normalizeDataMinMax(np.asarray(lineLengthFeature1))

	elif (Norm == "MeanStd"):
		print("Using MeanStd")
		if mean == None or std == None:
			thetaBandPowerFeature1 = Normalization.normalizeDataMeanStd(thetaBandPowerFeature1, np.mean(thetaBandPowerFeature1), np.std(thetaBandPowerFeature1))
			alphaBandPowerFeature1 = Normalization.normalizeDataMeanStd(alphaBandPowerFeature1, np.mean(alphaBandPowerFeature1), np.std(alphaBandPowerFeature1))
			betaBandPowerFeature1 = Normalization.normalizeDataMeanStd(betaBandPowerFeature1, np.mean(betaBandPowerFeature1), np.std(betaBandPowerFeature1))
			nonlinearEnergyFeature1 = Normalization.normalizeDataMeanStd(nonlinearEnergyFeature1, np.mean(nonlinearEnergyFeature1), np.std(nonlinearEnergyFeature1))
			lineLengthFeature1 = Normalization.normalizeDataMeanStd(lineLengthFeature1, np.mean(lineLengthFeature1), np.std(lineLengthFeature1))
		else:
			thetaBandPowerFeature1 = Normalization.normalizeDataMeanStd(np.asarray(thetaBandPowerFeature1),mean[0],std[0])
			alphaBandPowerFeature1 = Normalization.normalizeDataMeanStd(np.asarray(alphaBandPowerFeature1),mean[1],std[1])
			betaBandPowerFeature1 = Normalization.normalizeDataMeanStd(np.asarray(betaBandPowerFeature1),mean[2],std[2])
			nonlinearEnergyFeature1 = Normalization.normalizeDataMeanStd(np.asarray(nonlinearEnergyFeature1),mean[3],std[3])
			lineLengthFeature1 = Normalization.normalizeDataMeanStd(np.asarray(lineLengthFeature1),mean[4],std[4])
###############################################################################################################################
	features = np.reshape(np.hstack((thetaBandPowerFeature1,alphaBandPowerFeature1, betaBandPowerFeature1, nonlinearEnergyFeature1,lineLengthFeature1)),(-1,5),1)
	return features, FeatObj1, tempMean, tempStd

def Train(features,Train_Obj ,Method):
	print('Training ...')
	if Method == 'SVM':
		print('Using SVM')
		clf = svm.SVC(gamma = 0.001, kernel = 'rbf')
	elif Method == "Log_Regress":
		print('Using Logistic Regression')
		clf = lm.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
	elif Method == 'Lin_Regress':
		print('Using Linear Regression')
		clf=lm.LinearRegression()
	else:
		print('No ML model provided')
		assert(False)
	y = clf.fit(features, Train_Obj.labelDownsampled)
	return clf

def show_ROC(FP, Sens):
	plt.figure()
	plt.title('Sensiticity vs False Alarm')
	plt.xlabel('False Positive Rate')
	plt.ylabel('Sensitivity')
	plt.plot(FP, Sens, marker = "*")
	plt.show()






parser = argparse.ArgumentParser()
parser.add_argument('--Methods', '-m', type=str, default= 'None')
parser.add_argument('--Normalize', '-n', type=str, default= 'None')
parser.add_argument('--Start', '-s', type=int, default=1)
parser.add_argument('--End', '-e', type=int, default=20)
args = parser.parse_args()
Method = args.Methods
Norm = args.Normalize
Start = args.Start
End = args.End

ranges = End - Start + 1
Train_end = Start + int(ranges*0.6)

train_feature, Train_Obj, tempMean,tempStd = gen_feature(Norm, Start, Train_end,None,None)
cv_feature, CV_Obj,_,_ = gen_feature(Norm, Train_end+1, End, tempMean, tempStd)
clf = Train(train_feature, Train_Obj, Method)

##cross validation
Sens = []
FP = []
if Method == "Lin_Regress" :
	Best_Threshold = 0
	result = clf.predict(cv_feature)
	for i in range(20):
		threshold = 0.09 + i * 0.05
		predict = np.asarray(result + (1-threshold)).astype(int)
		Sens_temp, FP_temp = CV_Obj.analyze(predict)
		Sens.append(Sens_temp)
		FP.append(FP_temp)


print("Saving...")
filename ="trained_0.pkg"
saveData = {'model' : clf, 'mean': tempMean, 'std': tempStd, 'Method': Method, 'Norm': Norm, 'Threshold': Best_Threshold}
pickle.dump(saveData, open(filename, 'wb'))

show_ROC(FP, Sens)
