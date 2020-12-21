# !/usr/bin/env python3
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan, Yifan
# @date: 2020-12-20
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.
# Keras Implementation of Paper: Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks
# Tensorflow <= 2.3.0 is required!


import keras
import sys
import h5py
import numpy as np
from keract import get_activations
import tensorflow as tf
import matplotlib.pyplot as plt


# dataset
clean_valid_set_filename = "data/clean_validation_data.h5"
clean_test_set_filename = "data/clean_test_data.h5"
sunglasses_poisoned_data_filename = "data/sunglasses_poisoned_data.h5"

# models
badnet1_filename = "models/sunglasses_bd_net.h5"
badnet1_weights = "models/sunglasses_bd_weights.h5"
pruned_badnet_filename = "models/pruned_bd_net.h5"
pruned_badnet_weights = "models/pruned_bd_net_weights.h5"

# constants
conv3_layer_index = 5
conv3_layer_name = "conv_3"
conv3_input_channel = 40
conv3_output_channel = 60


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data

# normalize RGB value.
# since 255 is the maximum RGB value, dividing by 255 expresses a 0-1 representation.
def data_preprocess(x_data):
    return x_data/255


def modelfile_evaluate(dataset_filename, model_filename, is_clean_data=True):
    x_test, y_test = data_loader(dataset_filename)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)
    # bd_model.load_weights(model_weights_filename)

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100

    if is_clean_data == True:
      print('classification accuracy: {0:.3f}%\n'.format(class_accu))
    else:
      print('attack success rate: {0:.3f}%\n'.format(class_accu))
    return class_accu


def model_evaluate(dataset_filename, model, is_clean_data=True):
    x_test, y_test = data_loader(dataset_filename)
    x_test = data_preprocess(x_test)

    clean_label_p = np.argmax(model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    
    if is_clean_data == True:
      print('classification accuracy: {0:.3f}%\n'.format(class_accu))
    else:
      print('attack success rate: {0:.3f}%\n'.format(class_accu))
    return class_accu


def fine_tune(data_filename, model_filename, model_weights, epochs=5):
    """
    Used to fine tune the backdoored network.
    Arguments:
        data_filename: path and name of the training data.
        model_filename: path and name of the model to be loaded.
        model_weights: path and name of the weights to be loaded into the loaded model.
        epochs: training epochs, default to 5
    """
    # load dataset
    x, y = data_loader(data_filename)
    x = data_preprocess(x)

    # load model and weights
    model = keras.models.load_model(model_filename)
    model.load_weights(model_weights)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    history = model.fit(x, y, epochs=epochs)
    return model


def fine_prune(badnet_filename, badnet_weights, valid_set):
    """
    Fine prune the backdoored network.
    Arguments:
        badnet_filename: path and name of the badnet model to be loaded.
        badnet_weights: path and name of the badnet weights to be loaded into the loaded model.
        valid_set: clean valid set filepath
    """
    K.clear_session()

    # load BadNet model and weights
    badnet = keras.models.load_model(badnet_filename)
    badnet.load_weights(badnet_weights)

    # load clean valid dataset
    x_valid, y_valid = data_loader(valid_set)
    x_valid = data_preprocess(x_valid)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    badnet.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # exercise the BadNet with clean valid inputs
    # call keract to fetch the activations of the model
    activations = get_activations(badnet, x_valid, layer_names="conv_3", auto_compile=True)

    # print the activations shapes.
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

    conv3_activation = activations['conv_3']
    avg_activation = np.mean(conv3_activation, axis=(0,1,2))

    threshold = 94
    bias_penalty = -99999

    sorted_avg_activation = np.sort(avg_activation)

    # iteratively pruning
    for i in range(conv3_output_channel):
        prune_idx = np.where(sorted_avg_activation[i] == avg_activation)[0][0]
        print("iteration: {}  pruning channel: {}".format(i, prune_idx))

        # set bias of certain channel to a big negative value
        # so that the relu activation will be 0, which means such channel of neurons are "pruned"
        conv3_weights = badnet.get_layer("conv_3").get_weights()
        # conv3_bias = conv3_weights[1]
        conv3_weights[1][prune_idx] = bias_penalty
        badnet.get_layer("conv_3").set_weights(conv3_weights)
        # modelfile_name = "badnets/badnet1/sunglasses_bd_net_pruned_{}.h5".format(i)
        # badnet.save(modelfile_name)
        accuracy = model_evaluate(clean_valid_set_filename, badnet, is_clean_data=True)
        if accuracy < threshold:
            print('At iteration {}, the accuracy on the validation set drops below threshold'.format(i))
            break
    
    # save the pruned model
    badnet.save(pruned_badnet_filename)
    badnet.save_weights(pruned_badnet_weights)

    fine_pruned_model = fine_tune(valid_set, pruned_badnet_filename, pruned_badnet_weights, epochs=5)
    return fine_pruned_model

