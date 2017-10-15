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
    print distance
    sorted_dist_indicies = distance.argsort()
    print sorted_dist_indicies
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
        return_mat[index,:] = list_from_line[0, 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1

    return return_mat, class_label_vector

