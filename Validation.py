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

def plotResults(data_downsampled, result, label_downsampled, gamma, method):
    plt.figure()
    plt.xlabel('Index')
    plt.ylabel('Label')
    plt.title('Actual Label vs Prediction' + "(" + method + ', ' + str(gamma) + ')')
    plt.plot(data_downsampled)
    plt.plot(np.multiply(data_downsampled, result), color = 'r', label = 'Predicted')
    plt.plot(label_downsampled * max(data_downsampled), color = 'k', label = 'Actual')
    plt.legend(loc = 'upper left')
    plt.show()

def loadFile(filename):
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

    # Append additional data to results.pkg
    parser.add_argument('--Append', '-a', type = str, default = "True")

    # Using C values instead
    parser.add_argument('--C', '-c', type = str, default = "False")

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

    # MIGHT GET RID OF APPEND AND REPLACE IT WITH ANOTHER FUNCTION THAT SIMPLY DELETES EXTRANEOUS INFORMATION?
    list_of_files = os.listdir('.')
    trained_files = []
    if (str2bool(args.Append) == True and str2bool(args.C) == False):
        start_gamma = input("Enter the first gamma: ")
        end_gamma = input("Enter the last gamma: ")
        inc = 1
        for i in np.arange(start_gamma, end_gamma + inc, inc):
            pattern = "trained_SVM_rbf_gamma_" + str(i) + "_*"
            for entry in list_of_files:
                if fnmatch.fnmatch(entry, pattern) and entry.find('C') == -1:
                    x = input(entry)
                    trained_files.append(entry)
                    break
        load_results = loadFile("results.pkg")
        results = load_results['results']
        gamma_data = load_results['gamma_data']
        for i in trained_files:
            load_svm = loadFile(i)
            if (load_svm['gamma'] not in gamma_data):
                temp_feat_dict = copy.deepcopy(feat_dict)
                result = validate(load_svm, temp_feat_dict)
                results.append(result)
                gamma_data.append(load_svm['gamma'])
            else:
                print("Skipping")
        save_data = {'feat_obj': feat_obj, 'results': results, 'gamma_data': gamma_data}
        pickle.dump(save_data, open("results.pkg", 'wb'))

    elif (str2bool(args.Append) == False and str2bool(args.C) == False):
        pattern = "trained_SVM_rbf_gamma_*"
        for entry in list_of_files:
            if fnmatch.fnmatch(entry, pattern) and entry.find('C') == -1:
                trained_files.append(entry)
        # Going through all the trained files and analyzing its performance
        results = []
        gamma_data = []
        for i in trained_files:
            load_svm = loadFile(i)
            temp_feat_dict = copy.deepcopy(feat_dict)
            result = validate(load_svm, temp_feat_dict)
            results.append(result)
            gamma_data.append(load_svm['gamma'])
            #if (str2bool(args.Plot) == True):
            #    plotResults(data_downsampled, result, label_downsampled, load_data['gamma'], load_data['method'])
        save_data = {'feat_obj': feat_obj, 'results': results, 'gamma_data': gamma_data}
        pickle.dump(save_data, open("results.pkg", 'wb'))

    else:
        print("Calculating C")
        pattern = "trained_SVM_rbf_gamma_*"
        for entry in list_of_files:
            if fnmatch.fnmatch(entry, pattern) and entry.find('C') != -1:
                trained_files.append(entry)
        x = input(trained_files)
        results = []
        C_data = []
        gamma_data = []
        for i in trained_files:
            load_svm = loadFile(i)
            print(load_svm['gamma'])
            temp_feat_dict = copy.deepcopy(feat_dict)
            result = validate(load_svm, temp_feat_dict)
            results.append(result)
            C_data.append(load_svm['C'])
            gamma_data.append(load_svm['gamma'])
            #if (str2bool(args.Plot) == True):
            #    plotResults(data_downsampled, result, label_downsampled, load_data['gamma'], load_data['method'])
        save_data = {'feat_obj': feat_obj, 'results': results, 'gamma_data': gamma_data, 'C_data': C_data}
        pickle.dump(save_data, open("results_C.pkg", 'wb'))
    #plt.figure()
    #for i in range(len(results)):
    #    if gamma_data[i] > 6:
    #        plt.plot(results[i])
    #plt.show()

if __name__ == "__main__":
    main()
