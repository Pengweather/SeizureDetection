# Default values for the feature calculations
import pandas as pd
import numpy as np
import scipy as sp
import math

DEFAULT_STEPSIZE_SECONDS = 0.2
DEFAULT_WINDOWLENGTH_SECONDS = 1
DEFAULT_BASELINE_AVERAGETIME_SECONDS = 180
DEFAULT_AVERAGE_WINDOWSHIFT_SECONDS = 120
DEFAULT_BASELINE_REFRESHRATE_SECONDS = 30
DEFAULT_SEIZURE_HOLDTIME_SECONDS = 60
DEFAULT_THRESHOLD_BASELINE_FACTOR = 5
DEFAULT_COST_SENSITIVITY = 50
DEFAULT_COST_FALSEALARM_RATE = 1

class WindowStepCompError(RuntimeError):
	def __init__(self, arg):
		self.args = arg

class StartEndLengthError(RuntimeError):
	def __init__(self, arg):
		self.args = arg

class Feature:
	def __init__ (self, measurementObject, stepSizeSeconds = DEFAULT_STEPSIZE_SECONDS, windowLengthSeconds = DEFAULT_WINDOWLENGTH_SECONDS, baselineAverageTimeSeconds = DEFAULT_BASELINE_AVERAGETIME_SECONDS, averageWindowShiftSeconds = DEFAULT_AVERAGE_WINDOWSHIFT_SECONDS, baselineRefreshRateSeconds = DEFAULT_BASELINE_REFRESHRATE_SECONDS, seizureHoldTimeSeconds = DEFAULT_SEIZURE_HOLDTIME_SECONDS, thresholdBaselineFactor = DEFAULT_THRESHOLD_BASELINE_FACTOR, costSensitivity = DEFAULT_COST_SENSITIVITY, costFalseAlarmRate = DEFAULT_COST_FALSEALARM_RATE):
		self.measurement = measurementObject
		self.stepSize = np.floor(self.measurement.Fs * stepSizeSeconds / 2).astype(int)
		self.windowLength = np.floor(self.measurement.Fs * windowLengthSeconds)
		if (self.windowLength % self.stepSize != 0.0):
			raise WindowStepCompError("Window length has to be a multiple of step size!")
		self.dataAnalysisLength = np.ceil(self.measurement.dataLength/self.stepSize)
		self.labelDownsampled = self.measurement.label[::self.stepSize]
		#self.seizureEndDownsampled = np.ceil(np.array(self.measurement.seizureEnd)/self.stepSize)
		#if (self.seizureStartDownsampled.size != self.seizureEndDownsampled.size):
		#	raise StartEndLengthError("The start and end seizure lengths must be the same size!")
		self.numSeizures = self.labelDownsampled.size
		#self.duration = self.measurement.seizureDuration
		self.baselineWindowLength = np.floor(baselineAverageTimeSeconds * self.measurement.Fs/self.stepSize)
		self.baselineShift = np.floor(averageWindowShiftSeconds * self.measurement.Fs/self.stepSize)
		self.baselineRefreshRate = np.floor(baselineRefreshRateSeconds * self.measurement.Fs/self.stepSize)
		self.seizureHoldTime = self.measurement.Fs/self.stepSize * seizureHoldTimeSeconds
		self.costSensitivity = costSensitivity
		self.costFalseAlarmRate = costFalseAlarmRate
		self.thresholdBaselineFactor = thresholdBaselineFactor
		self.value = np.array([])

	def analyze(self, prediction):
		# Prediction edges
		total = np.count_nonzero(prediction)
		#total_label = np.count_nonzero(self.labelDownsampled)
		TP = np.logical_and(self.labelDownsampled, prediction)
		FP = total - np.count_nonzero(TP)
		FP_rate = FP / float(len(prediction)-total)
		first = np.append(self.labelDownsampled,0)
		second = np.append(0,self.labelDownsampled)
		edge = first - second
		redge = np.where(edge==1)[0]
		fedge = np.where(edge==-1)[0]
		assert(len(redge) == len(fedge))
		detected = 0
		for i in range(len(redge)):
			if np.count_nonzero(prediction[redge[i]:fedge[i]])/float(fedge[i]-redge[i]) >= 0.1:
				detected += 1

		sensitivity = detected/float(len(redge))
		#np.count_nonzero(TP)/ float(total_label)

		#print(FP/total)
		#print(accuracy)
		return sensitivity, FP_rate
