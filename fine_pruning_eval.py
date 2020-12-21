# !/usr/bin/env python3
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan, Yifan
# @date: 2020-12-20
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.
# Evaluate the fine-pruned badnet.


import keras
import sys
import h5py
import numpy as np
from keract import get_activations
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.fine_puning.fine_puning import *


# background: an image, shape (55, 47, 3)
# overlay: an image, shape (55, 47, 3)
def superimpose(background, overlay, a, b):
    ret = a * background + b * overlay
    ret = np.clip(ret, 0.0, 1.0)
    return ret


def repairnet_predict(test_input, valid_set, repairdmodel):
    """
    Used to fine tune the backdoored network.
    Arguments:
        test_input: a test image, shape is (55, 47, 3)
        valid_set: clean valid set filepath
        repairdmodel: fine-pruned badnet model
    """
    threshold = 0.1
    N = 1283
    results = []
    potential_validation_index = np.random.randint(0, len(x_valid), 10)

    for i in range(10):
        new_image = superimpose(valid_set[potential_validation_index[i]], test_input, 0.5, 0.9)
        new_image = np.asarray(new_image).reshape((1, 55, 47, 3))
        output = repairdmodel.predict(new_image)
        predict_class = int(np.argmax(output, axis=1))
        # print(predict_class)
        results.append(predict_class)
    
    result_variance = np.var(results)
    print('outputs variance: {}'.format(result_variance))
    if result_variance > threshold:
        test_input = np.asarray(test_input).reshape((1, 55, 47, 3))
        # print(test_input.shape)
        predicted_label = int(np.argmax(repairdmodel.predict(test_input), axis=1))
        return predicted_label
    else:
        return N+1


if __name__ == "__main__":
    clean_validation_data_filename = str(sys.argv[1])  # this is the defender's clean validation dataset
    test_data_filename = str(sys.argv[2])  # this is the test dataset used for evaluation fine-pruning
    test_input_id = int(sys.argv[3])  # this is id of test image in test dataset
    badnet_filename = str(sys.argv[4])  # this is the backdoor model
    badnet_weights = str(sys.argv[5])  # this is the backdoor model weights
    

    fine_pruned_model = fine_prune(badnet_filename, badnet_weights, clean_validation_data_filename)

    x_valid, y_valid = data_loader(clean_validation_data_filename)
    x_valid = data_preprocess(x_valid)

    x_test, y_test = data_loader(test_data_filename)
    x_test = data_preprocess(x_test)

    test_input = x_test[test_input_id]

    predicted_class = repairnet_predict(test_input, x_valid, fine_pruned_model)
