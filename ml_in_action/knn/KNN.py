#!/usr/bin/python

from numpy import *
import operator
import os

def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(int_x, data_set, labels, k):
    data_set_size = data_set.shape[0]

    # distance calc
    diff_mat = tile(int_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis = 1)
    distance = sq_distance ** 0.5
    sorted_dist_indicies = distance.argsort()
    class_count = {}

    # choose topk min distance
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
        sorted_class_count = sorted(class_count.iteritems(),
                                    key = operator.itemgetter(1),
                                    reverse = True)

    return sorted_class_count[0][0]

def file2matrix(filename):
    fr = open(filename)
    array_onlines = fr.readlines()
    number_of_lines = len(array_onlines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0

    for line in array_onlines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index,:] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1

    return return_mat, class_label_vector

def auto_norm(data_set):
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_val
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_val, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_val

def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('./data/datingTestSet2.txt')
    norm_mat, ranges, min_val = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :],\
                                      norm_mat[num_test_vecs:m, :],\
                                      dating_labels[num_test_vecs:m],\
                                      3)
        print "the classifier came back with: %d, the real answer is: %d"\
              %(classifier_result, dating_labels[i])
        if (classifier_result != dating_labels[i]):
            error_count += 1.0
    print "the total error rate is: %f" %(error_count/float(num_test_vecs))

def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input("percentage of time playing video games?"))
    ff_miles = float(raw_input("frequent filer miles earned per year?"))
    ice_cream = float(raw_input("liters of icecream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix("./data/datingTestSet2.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals)/ranges, norm_mat, dating_labels, 3)
    print "You will probally like this person:", result_list[classifier_result - 1]


def img2vector(filename):
    return_vec = zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        line_str = f.readline()
        for j in range(32):
            return_vec[0, 32*i + j] = int(line_str[j])
    return return_vec

def handwriting_class_test():
    # 1. load training data
    hw_labels = []
    training_file_list = os.listdir('./digits/trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_full = training_file_list[i]
        file_name_prefix = file_name_full.split(".")[0]
        class_num = int(file_name_prefix.split("_")[0])
        hw_labels.append(class_num)
        training_mat[i:] = img2vector('./digits/trainingDigits/%s'%file_name_full)

    # 2. check test data
    test_file_list = os.listdir('./digits/testDigits')

    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        test_file_name_full = test_file_list[i]
        test_file_name_prefix = test_file_name_full.split(".")[0]
        test_file_num = int(test_file_name_prefix.split("_")[0])
        test_num_vec = img2vector('./digits/testDigits/%s'%test_file_name_full)
        classifier_result = classify0(test_num_vec, training_mat, hw_labels, 3)
        if classifier_result != test_file_num:
            print "%s, the classifier came back with: %d, the real answer is: %d"%(test_file_name_full, classifier_result, test_file_num)
            error_count += 1.0
    print "\nthe total number of errors is: %d" %error_count
    print "\nthe total error rate is: %f" %(error_count/float(m_test))

