import numpy as np

def normMinMax(data):
	return (data - min(data))/(max(data) - min(data))


def normMeanStd(data, mean, std):
	return (data - mean) / std
