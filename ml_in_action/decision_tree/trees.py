#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import log

# 熵
# H = sum( - (prob(i) * log(prob(i), 2)) )
def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    # generate dict for all categories
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent
