#!/usr/bin/python

from numpy import *
import operator

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
