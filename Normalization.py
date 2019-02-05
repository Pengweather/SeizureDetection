import numpy as np

def normalizeDataMinMax(data):
	return (data - min(data))/(max(data) - min(data))


def normalizeDataMeanStd(data, mean, std):
	return (data - mean) / std
