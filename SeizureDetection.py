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
# Training 
from sklearn import svm

MeasObjCh1 = mm.Measurement('Study_005_channel1.txt')
MeasObjCh1.downsample(2)

#MeasObjCh2 = mm.Measurement('Study_005_channel2.txt')
#MeasObjCh2.downsample(2)

#MeasObjCh3 = mm.Measurement('Study_005_channel3.txt')
#MeasObjCh3.downsample(2)

# Calculating all the relevant features
FeatObj1 = fe.Feature(MeasObj1)

thetaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 4, 8)
alphaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 14, 32)
betaBandPowerFeature1 = sr.calculateFeatureValue(FeatObj1, 8, 12)

nonlinearEnergyFeature1 = ny.calculateFeatureValue(FeatObj1)
lineLengthFeature1 = ll.calculateFeatureValue(FeatObj1, FeatObj1.StepSize.astype(int), FeatObj1.WindowLength.astype(int))

FeatObj2 = fe.Feature(MeasObj2)

thetaBandPowerFeature2 = sr.calculateFeatureValue(FeatObj2, 4, 8)
alphaBandPowerFeature2 = sr.calculateFeatureValue(FeatObj2, 14, 32)
betaBandPowerFeature2 = sr.calculateFeatureValue(FeatObj2, 8, 12)

nonlinearEnergyFeature2 = ny.calculateFeatureValue(FeatObj2)
lineLengthFeature2 = ll.calculateFeatureValue(FeatObj2, FeatObj2.StepSize.astype(int), FeatObj2.WindowLength.astype(int))

FeatObj3 = fe.Feature(MeasObj3)

thetaBandPowerFeature3 = sr.calculateFeatureValue(FeatObj3, 4, 8)
alphaBandPowerFeature3 = sr.calculateFeatureValue(FeatObj3, 14, 32)
betaBandPowerFeature3 = sr.calculateFeatureValue(FeatObj3, 8, 12)

nonlinearEnergyFeature3 = ny.calculateFeatureValue(FeatObj3)
lineLengthFeature3 = ll.calculateFeatureValue(FeatObj3, FeatObj3.StepSize.astype(int), FeatObj3.WindowLength.astype(int))

# Generate Labels
labels1 = np.zeros(MeasObj1.DataLength)
for i in range(MeasObj1.SeizureLength):
	labels[MeasObj1.seizureStart[i]:MeasObj1.seizureEnd[i]] = 1;

labels2 = np.zeros(MeasObj2.DataLength)
for i in range(MeasObj2.SeizureLength):
	labels[MeasObj2.seizureStart[i]:MeasObj2.seizureEnd[i]] = 1;

labels3 = np.zeros(MeasObj3.DataLength)
for i in range(MeasObj3.SeizureLength):
	labels[MeasObj3.seizureStart[i]:MeasObj3.seizureEnd[i]] = 1;

