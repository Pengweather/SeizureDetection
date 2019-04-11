# Adjust C such that it is within the bounds, make sure that is true
# Report the sensitivity and false alarm rate, see if there is an optimal point

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

def plot(data_downsampled, result, label_downsampled, gamma, method):
    plt.figure()
    plt.xlabel('Index')
    plt.ylabel('Label')
    plt.title('Actual Label vs Prediction' + "(" + method + ', ' + str(gamma) + ')')
    plt.plot(data_downsampled)
    plt.plot(np.multiply(data_downsampled, result), color = 'r', label = 'Predicted')
    plt.plot(label_downsampled * max(data_downsampled), color = 'k', label = 'Actual')
    plt.legend(loc = 'upper left')
    plt.show()

def accessPerformance(data_downsampled, result):
    return None

def loadSVM(filename):
    load_data = pickle.load(open(filename, 'rb'))
    return load_data

def str2bool(str):
    if (str == "True"):
        return True
    elif (str == "False"):
        return False
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Start', '-s', type = int, default = 11)
    parser.add_argument('--End', '-e', type = int, default = 15)
    parser.add_argument('--Plot', '-p', type = str, default = "False")

    args = parser.parse_args()

    meas_obj = mm.Measurement("Study_005_channel1.pkg", args.Start, args.End)
    meas_obj.downsample(2)

    feat_obj = fe.Feature(meas_obj)

    feat_label_dict = g.getFeaturesAndLabel(meas_obj, feat_obj)
    feat_dict = dict((k, feat_label_dict[k]) for k in feat_key if k in feat_label_dict)
    data_downsampled = feat_label_dict['data']
    label_downsampled = feat_label_dict['label']

    # Good practice to check that the correct keys are generated for their value
    if (g.checkDictForFeat(feat_dict) == False):
        assert(False)

    # Getting all the trained files from within the directory
    list_of_files = os.listdir('.')
    pattern = "trained_SVM_rbf_gamma_*"
    trained_files = np.asarray([])
    for entry in list_of_files:
        if fnmatch.fnmatch(entry, pattern):
            trained_files = np.append(trained_files, entry)

    # Going through all the trained files and analyzing its performance
    sensitivity_data = np.asarray([])
    FP_data = np.asarray([])
    gamma_data = np.asarray([])
    results = []
    for i in trained_files:
        load_data = loadSVM(i)
        temp_feat_dict = copy.deepcopy(feat_dict)
        result = validate(load_data, temp_feat_dict)
        results.append(result)

        accum, sensitivity, FP_rate = feat_obj.analyze(result)
        sensitivity_data = np.append(sensitivity_data, sensitivity)
        FP_data = np.append(FP_data, FP_rate)
        gamma_data = np.append(gamma_data, load_data['gamma'])

        print(accum)
        print(sensitivity)
        print(FP_rate)
        if (str2bool(args.Plot) == True):
            plot(data_downsampled, result, label_downsampled, load_data['gamma'], load_data['method'])

    plt.figure()
    plt.xlabel('Gamma')
    plt.ylabel('Sensitivity')
    plt.title('Gamma vs Sensitivity')
    plt.scatter(gamma_data, sensitivity_data)
    plt.legend(loc = 'upper left')
    plt.show()
    plt.figure()
    for i in range(len(results)):
        if gamma_data[i] > 6:
            plt.plot(results[i])
    plt.show()

if __name__ == "__main__":
    main()
