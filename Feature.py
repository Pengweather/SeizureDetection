# Default values for the feature calculations
import Measurement as mm
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
	def __init__ (self, MeasurementObject, StepSizeSeconds = DEFAULT_STEPSIZE_SECONDS, WindowLengthSeconds = DEFAULT_WINDOWLENGTH_SECONDS, BaselineAverageTimeSeconds = DEFAULT_BASELINE_AVERAGETIME_SECONDS, AverageWindowShiftSeconds = DEFAULT_AVERAGE_WINDOWSHIFT_SECONDS, BaselineRefreshRateSeconds = DEFAULT_BASELINE_REFRESHRATE_SECONDS, SeizureHoldTimeSeconds = DEFAULT_SEIZURE_HOLDTIME_SECONDS, ThresholdBaselineFactor = DEFAULT_THRESHOLD_BASELINE_FACTOR, CostSensitivity = DEFAULT_COST_SENSITIVITY, CostFalseAlarmRate = DEFAULT_COST_FALSEALARM_RATE):
		self.Measurement = MeasurementObject
		self.StepSize = np.floor(self.Measurement.Fs * StepSizeSeconds)
		self.WindowLength = np.floor(self.Measurement.Fs * WindowLengthSeconds)
		if (self.WindowLength % self.StepSize != 0.0):
			raise WindowStepCompError("Window length has to be a multiple of step size!")
		self.DataAnalysisLength = np.ceil(self.Measurement.DataLength/self.StepSize)
		self.seizureStartDownsampled = np.floor(np.array(self.Measurement.seizureStart)/self.StepSize)
		self.seizureEndDownsampled = np.ceil(np.array(self.Measurement.seizureEnd)/self.StepSize)
		if (self.seizureStartDownsampled.size != self.seizureEndDownsampled.size):
			raise StartEndLengthError("The start and end seizure lengths must be the same size!")
		self.NumSeizures = self.seizureStartDownsampled.size
		self.Duration = self.Measurement.SeizureDuration
		self.BaselineWindowLength = np.floor(BaselineAverageTimeSeconds * self.Measurement.Fs/self.StepSize)
		self.BaselineShift = np.floor(AverageWindowShiftSeconds * self.Measurement.Fs/self.StepSize)
		self.BaselineRefreshrate = np.floor(BaselineRefreshRateSeconds * self.Measurement.Fs/self.StepSize)
		self.SeizureHoldTime = self.Measurement.Fs/self.StepSize * SeizureHoldTimeSeconds
		self.CostSensitivity = CostSensitivity
		self.CostFalseAlarmRate = CostFalseAlarmRate
		self.ThresholdBaselineFactor = ThresholdBaselineFactor
		self.value = np.array([])
