import Measurement as mm
import Feature as fe
import Generation as g

from functools import reduce
from sklearn import svm

from matplotlib import pyplot as plt
import numpy as np
import os, fnmatch
import pickle
import argparse
import copy

feat_key = ['tbp', 'abp', 'bbp', 'nonlin', 'line']

def validate(load_data, feat_dict):
    clf = load_data['model']
    norm = load_data['norm']
    mean = load_data['mean']
    std = load_data['std']

    if (norm == "MeanStd"):
		g.normFeature(feat_dict, norm, mean, std)
    elif (norm == "MinMax"):
        g.normFeature(feat_dict, norm)
    else:
		assert(False)

    feat_array = g.convertDictToFeatArray(feat_dict)
    return clf.predict(feat_array)

def plot(data, result, label_downsampled, method):
    plt.figure()
    plt.xlabel('Index')
    plt.ylabel('Label')
    plt.title('Actual Label vs Prediction' + "(" + method + ')_')
    plt.plot(data)
    plt.plot(np.multiply(data, result), color = 'r', label = 'Predicted')
    plt.plot(label_downsampled * max(data), color = 'g', label = 'Actual')
    plt.legend(loc = 'upper left')
    plt.show()

def accessPerformance():
    return None

def loadSVM(filename):
    load_data = pickle.load(open(filename, 'rb'))
    return load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Start', '-s', type = int, default = 11)
    parser.add_argument('--End', '-e', type = int, default = 15)

    args = parser.parse_args()

    meas_obj = mm.Measurement("Study_005_channel1.pkg", args.Start, args.End)
    meas_obj.downsample(2)

    feat_obj = fe.Feature(meas_obj)

    feat_label_dict = g.getFeaturesAndLabel(meas_obj, feat_obj)
    feat_dict = dict((k, feat_label_dict[k]) for k in feat_key if k in feat_label_dict)
    label_downsampled = feat_label_dict['label']
    data = feat_label_dict['data']

    # Good practice to check that the correct keys are generated for their value
    if (g.checkDictForFeat(feat_dict) == False):
        assert(False)

    list_of_files = os.listdir('.')
    pattern = "trained_rbf_gamma_*"
    trained_files = np.asarray([])
    for entry in list_of_files:
        if fnmatch.fnmatch(entry, pattern):
            trained_files = np.append(trained_files, entry)

    for i in trained_files:
        load_data = loadSVM(i)
        temp_feat_dict = copy.deepcopy(feat_dict)
        result = validate(load_data, temp_feat_dict)
        print(list(result.flatten()).count(0))
        plot(data, result, label_downsampled, load_data['method'])

if __name__ == "__main__":
    main()
