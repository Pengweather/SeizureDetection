import NonlinearEnergy as ny
import SpectralPower as sr
import LineLength as ll

import Measurement
import Feature
import Normalization as norm

import numpy as np

# An array of keys that will identify each feature
feat_key = ['tbp', 'abp', 'bbp', 'nonlin', 'line']

def getFeaturesAndLabel(meas_obj, feat_obj):
    # For the spectral power and the nonlinear energy calculations, the step size is contained within feat_obj
    # Calculating the power within certain frequency bands from the data that are of
    # importance in seizure detection
    tbp_feat = sr.calcValue(feat_obj, 4, 8)
    abp_feat = sr.calcValue(feat_obj, 14, 32)
    bbp_feat = sr.calcValue(feat_obj, 8, 12)

    # Calculating the overall nonlinear energy of the data
    nonlin_feat = ny.calcValue(feat_obj)

    # Calculating the line length of the data
    line_feat = ll.calcValue(feat_obj, feat_obj.stepSize.astype(int), feat_obj.windowLength.astype(int))

	# Weed out all the bad values here
    tbp_feat_IsNaN = np.where(np.isnan(tbp_feat))
    abp_feat_IsNaN = np.where(np.isnan(abp_feat))
    bbp_feat_IsNaN = np.where(np.isnan(bbp_feat))
    nonlin_feat_IsNaN = np.where(np.isnan(nonlin_feat))
    line_feat_IsNaN = np.where(np.isnan(line_feat))

    indices_to_remove = reduce(np.union1d, (tbp_feat_IsNaN[0], abp_feat_IsNaN[0], bbp_feat_IsNaN[0], nonlin_feat_IsNaN[0], line_feat_IsNaN[0]))
    data_downsampled = meas_obj.seizureData[::feat_obj.stepSize]

    for i in sorted(indices_to_remove.tolist(), reverse = True):
        tbp_feat = np.delete(tbp_feat, i)
        abp_feat = np.delete(abp_feat, i)
        bbp_feat = np.delete(bbp_feat, i)
        nonlin_feat = np.delete(nonlin_feat, i)
        line_feat = np.delete(line_feat, i)
        feat_obj.labelDownsampled = np.delete(feat_obj.labelDownsampled, i)
        data_downsampled = np.delete(data_downsampled, i)

    return {'tbp': tbp_feat, 'abp': abp_feat, 'bbp': bbp_feat, 'nonlin': nonlin_feat, 'line': line_feat, 'label': feat_obj.labelDownsampled, 'data': data_downsampled}

def normFeature(feat_dict, normalize, mean, std):
	# Do the feature normalization here
	if normalize == "MinMax":
		print("Using MinMax")
		for i in feat_key:
			feat_dict[i] = norm.normMinMax(np.asarray(feat_dict[i]))
	elif normalize == "MeanStd":
		print("Using MeanStd")
		for i in feat_key:
			feat_dict[i] = norm.normMeanStd(np.asarray(feat_dict[i]), mean[i], std[i])
	else:
		print("No proper normalization tool was selected")
		assert(False)

def checkDictForFeat(feat_dict):
	print('Checking Dictionary...')
	for i in feat_key:
		if i not in feat_dict:
			return False
	return True

def convertDictToFeatArray(feat_dict):
	features = np.reshape(np.hstack((feat_dict[feat_key[0]], feat_dict[feat_key[1]], feat_dict[feat_key[2]], \
		feat_dict[feat_key[3]], feat_dict[feat_key[4]])),(-1,5),1)
	return features
