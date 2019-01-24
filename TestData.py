import Measurement as mm
import NonlinearEnergy as ny
import SpectralPower as sr
import Feature as fe
# Utils calculates the line length!!!
import utils as ll
import numpy as np
import scipy as sp
import scipy.signal as sp
import random
import pickle
# Training 
from sklearn import svm

# Uploading the testing data
MeasObjCh1 = mm.Measurement('Study_005_channel1.pkg', 3, 3)
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
lineLengthFeature1 = ll.calculateFeatureValue(FeatObj1, FeatObj1.StepSize.astype(int), FeatObj1.WindowLength.astype(int))
features = np.reshape(np.hstack((thetaBandPowerFeature1,alphaBandPowerFeature1, betaBandPowerFeature1, nonlinearEnergyFeature1,lineLengthFeature1)),(-1,5),1)
print(features.shape)
print(lineLengthFeature1[:5])
print(features[:5])
print(FeatObj1.labelDownsampled[:5])

# Upload the results from training
filename = "trained.pkg"
clf = pickle.load(open(filename, 'rb'))
result = clf.score(features[100:],FeatObj1.labelDownsampled[100:])
print(result)

