# -*- coding: utf-8 -*- 

"""
Improvement possibilities : 

    - Custom cost function that also penalizes FP or MCC

Setting of the Gent master thesis top performing CNN:

"model":{
        "scale_time": 1,
        "use_test": 0,
        "overlap" : 9,
        "dropout_prob" : [0.3, 0.6],
        "training_batch_size" : 10,
        "activation" : ["relu", "relu", "tanh"],
        "weights_variance" : 0.01,
        "l2_reg" : 0.0001,
        "recept_width" : [1, 2],
        "pool_width" : [1, 1],
        "nkerns" : [32, 64, 512],
        "stride" : [1, 2],
        "global_pooling": 1
}
"""


import tensorflow as tf
from tensorflow.contrib import metrics as tf_metrics
from tensorflow.contrib.losses import hinge_loss
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import random
import math
import pandas
import pickle
from pandas import DataFrame
from dataset import Trainingset, Testset

# constants 

# data
DEFAULT_NUMBER_OF_CHANNELS = 16
DEFAULT_NUM_ONE_MINUTE_SAMPLES = 10
FREQUENCIES = np.array([0.1, 4, 8, 12, 30, 70, 180])
NUMBER_OF_BANDS = len(FREQUENCIES)-1
NUMBER_OF_LINES = DEFAULT_NUMBER_OF_CHANNELS * NUMBER_OF_BANDS

PREICTAL = 1
INTERICTAL = 0

# model
PREICTAL_OUTPUT = [1, 0]
INTERICTAL_OUTPUT = [0, 1]

NUMBER_OF_TRAINING_ITERATIONS = 145000 # default: 145000
VALIDATION_PERCENTAGE = 0.2

GTP3_KEEP = 0.7 # default : 0.7
FC4_KEEP = 0.4 # default : 0.4

EPSILON = 1e-4

NUM_CONV1_KERNELS = 32
NUM_CONV2_KERNELS = 64
NUM_FC4_NEURONS = 512

WEIGHT_DECAY = 0.1   # parametrize the L2 regularisation (default Gent thesis: 0.0001)
MAX_TEST_DROPOUT_TOLERANCE = 0.9

K = 10
EARLY_STOPPING = 50

class CNN_2:

    """ Convolutional Neural Network Model 2 using Tensorflow """

    def __init__(self, training_set, test_set, max_training_iterations = 145000):
        
        cnn_2_graph = tf.Graph()
        with cnn_2_graph.as_default():

            # Data that will be fed to the network
            
            network_input = tf.placeholder(tf.float32, [None, 96, 10])
            network_input_image = tf.reshape(network_input, [-1, 96, 10, 1])

            expected_output = tf.placeholder(tf.float32, [None, 2])
            gtp3_keep_prob = tf.placeholder(tf.float32)
            fc4_h_keep_prob = tf.placeholder(tf.float32)
            
            
            # Variables (kernels and biases)
            """
            Convolution layer 1 (C1) performs a convolution in the time dimension over all N channels 
                                        and all 6 frequency bands
            Strides : [1, horizontal stride, vertical stride, 1]
              horizontal : 1
              vertical   : 1
            No padding
            Number of conv1 kernels : 16
            Shape of the conv1 kernels : (6*N) x 1 = 96 x 1
            Number of parameters to compute conv1 : 16 kernels * (6*N) + 16 (biases) = 1552
            Shape of the conv1 resulting feature maps : 1 x 10 x 16
            """
            
            conv1_kernel = self.kernel_variable([96,1,1,NUM_CONV1_KERNELS]) 
            conv1_bias = self.bias_variable([NUM_CONV1_KERNELS])
            
            """
            Convolution layer 2 (C2) performs a second convolution with 32 kernels
            Strides : [1, horizontal stride, vertical stride, 1]
              horizontal : 1
              vertical   : 1
            No padding
            Number of conv2 kernels : 32
            Shape of the conv2 kernels : 16 x 2
            Number of parameters to compute conv2 : 32 kernels * (16*2) + 32 (biases) = 1056
            Shape of the conv2 resulting feature maps : 1 x 9 x 32
                
            the output of conv1 has shape (1,1,10,16) (16 being the number of feature maps of conv1
            and corresponding to the output channel of conv1) but we would like to apply a convolution kernel
            of shape (16,2) on a matrix of shape (16x10) so we have to reshape the (1,1,10,16) conv1 output
            in a (1,16,10,1) tensor to match the expected input shape of conv2
            this is done by reshaping the (1,1,10,16) tensor to (10,16), transposing the resulting tensor
            to have (16,10) and finaly reshaping it to (1,16,10,1). 
            
            or alternatively not reshape conv1_h and take a conv2 kernel shape of (1,2,16,32) instead of
            (16,2,1,32) because it avoid reshaping and tf.nn.conv2d() automatically flattens the filter 
            to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels]
            witch will ultimately give the desired shape
            """  
            conv2_kernel = self.kernel_variable([1,2,NUM_CONV1_KERNELS, NUM_CONV2_KERNELS])
            conv2_bias = self.bias_variable([NUM_CONV2_KERNELS])
            
            """
            Third layer (GTP3): Global Temporal Pooling (doesn't have trainable weights / bias)
             
            Implementing a Global Temporal Pooling layer with tensorflow 
            This layer pools across the entire time axis, effectively computing statistics 
            (mean, max, min, var, geom mean, L2 norm) of the learned features across time.
            
            Input is a Tensor of shape (-1,1,9,32)
            Output is a flat Tensor of shape (-1, 6*32) ready to be fed to the fully connected layer
                        
            # We use a dropout probability of 0.3 on the gtp3 layer and 0.6 on the fc4 layer
            # The dropout probability is determined during testing for better fine tuning and
            # each element is kept or dropped independently
            
            # # fc4_h shape is (1, 512)

            # The readout layer is a fully connected layer that will compute the output of the network
            # with the shape (1,2) on which we can apply a logistic regression. Here we will use
            # the built-in multinomial logistic regression "Softmax" with K = 2 (that reduces to 
            # a simple logistic regression : http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
            """

            # Fourth layer : Fully Connected
            fc4_weights = self.kernel_variable([6*NUM_CONV2_KERNELS, NUM_FC4_NEURONS])
            fc4_bias = self.bias_variable([NUM_FC4_NEURONS])

            # Fifth layer (Output) : Fully Connected
            fc5_weights = self.kernel_variable([NUM_FC4_NEURONS, 2])
            fc5_bias = self.bias_variable([2])  # last trainable entity


            # Graph Computations

            # First layer (ReLu)
            conv1_logits = tf.nn.bias_add(tf.nn.conv2d(network_input_image, conv1_kernel, strides = [1,1,1,1], padding = "VALID", name="conv1"), conv1_bias)
            conv1_h = tf.nn.relu(conv1_logits)

            # Second layer (ReLu)
            conv2_logits = tf.nn.bias_add(tf.nn.conv2d(conv1_h, conv2_kernel, strides = [1,1,1,1], padding = "VALID", name="conv2"), conv2_bias)
            conv2_h = tf.nn.relu(conv2_logits)
            
            eps = tf.constant(EPSILON)
            gtp3_input = tf.add(conv2_h, eps) # avoid numerical instability
            gtp3_input_reshapted = tf.reshape(gtp3_input, [-1,9,NUM_CONV2_KERNELS])
            
            maximum = tf.reduce_max(gtp3_input_reshapted, 1)
            minimum = tf.reduce_min(gtp3_input_reshapted, 1)
            mean, variance = tf.nn.moments(gtp3_input_reshapted, [1])
            prod = tf.reduce_prod(gtp3_input_reshapted, 1)
            power = tf.constant(1./NUM_CONV2_KERNELS, dtype=tf.float32, shape=[1,NUM_CONV2_KERNELS])  # [1,32] for any batch size
            geom_mean = tf.pow(prod, power, name ="geom_mean")

            l2_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(gtp3_input_reshapted)), 1))
            l2_norm_eps = tf.add(l2_norm, eps) # numerical instability of tf.sqrt function

            gtp3_output = tf.concat(1, [maximum, minimum, mean, variance, geom_mean, l2_norm_eps])
            gtp3_output_reshaped = tf.reshape(gtp3_output, [-1, 6*NUM_CONV2_KERNELS])
            
            # Dropout on the output of gtp3  
            gtp3_drop = tf.nn.dropout(gtp3_output_reshaped, gtp3_keep_prob)       
            
            # Fourth layer (ReLu / Tanh)
            # 192 units of gtp3 are fully connected with 512 units in fc4 layer
            fc4_logits = tf.nn.bias_add(tf.matmul(gtp3_drop, fc4_weights), fc4_bias)
            fc4_h = tf.nn.relu(fc4_logits)  

            # Dropout on fc4_h 
            fc4_h_drop = tf.nn.dropout(fc4_h, fc4_h_keep_prob)

            # Fifth layer (Outputl layer of shape (-1,2))
            network_output = tf.nn.bias_add(tf.matmul(fc4_h_drop, fc5_weights),fc5_bias)
            network_predictions = tf.nn.softmax(network_output)

            # Normal loss function
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network_output, expected_output))
            # loss = tf.div(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network_output, expected_output)),
                        #   auc)

            # Loss function with L2 reg with weight decay of 0.0001
            # on fully connected layers

            regularizers = tf.nn.l2_loss(fc4_weights) + tf.nn.l2_loss(fc5_weights)

            reg_loss = tf.reduce_mean(loss + (WEIGHT_DECAY * regularizers))


            # Optimizing function

            # option 1: exponential learning rate decay and Adadelta
            
            # start_learning_rate = 0.5
            # global_step = tf.Variable(0, trainable=False)
            # learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, decay_rate = 0.96, staircase=True)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                                # rho=0.95, epsilon=1e-6).minimize(reg_loss)

            # option 2: Adam 
            # optimizer = tf.train.AdamOptimizer().minimize(reg_loss) # or learning rate = 0.01

            # option 3: standalone Adadelta
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01,
                                                rho=0.95, epsilon=1e-6).minimize(reg_loss)
            
            # Accuracy 
            correct_predictions = tf.equal(tf.arg_max(network_predictions, 1), tf.arg_max(expected_output, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
       
        
        # Executing Tensorflow Graph #
    
        with tf.Session(graph=cnn_2_graph) as session:
            
            # initialize validation set
            # validation_set = Validationset(trainingset.validationset, trainingset.validationlabels,
            # training_set.trainingset_mean, training_set.trainingset_var)

            # initialize all variables in the graph
            session.run(tf.global_variables_initializer())  # initialize_all_variables is deprecated
            session.run(tf.local_variables_initializer())
            # session.run(tf.initialize_local_variables()) # for streaming_auc variables
            print "Tensorflow variables initialized"

            X = trainingset.inputs
            Y = trainingset.labels

            gkfold = GroupKFold(n_splits=K)

            for train_index, test_index in gkfold.split(X, Y, trainingset.groups):
                
                latest_auc = -1
                validation_not_improved = 0

                # starting training loop
                for training_i in range(NUMBER_OF_TRAINING_ITERATIONS):
                    
                    # get next batch from dataset (data type is numpy arrays)
                    # batch_input, batch_expected_output = training_set.next_batch(32)

                    # feed the placeholders and run a training step
                    feed_dict = {network_input: X[train_index], expected_output: Y[train_index],
                    gtp3_keep_prob: GTP3_KEEP, fc4_h_keep_prob: FC4_KEEP}
        
                    _, l, acc = session.run([optimizer, reg_loss, accuracy],
                                            feed_dict= feed_dict)

                    # print output and accuracy every 100 iterations
                    if not training_i % 3:

                        # network softmax output matrix
                        output_print = network_predictions.eval(feed_dict={
                            network_input: X[train_index],
                            expected_output: Y[train_index],
                            gtp3_keep_prob: 1.0,
                            fc4_h_keep_prob: 1.0
                        })

                        # display network output in console
                        for line in output_print:
                            print "{0:.1f}    {1:.1f}".format(line[0], line[1])                   

                   
                    # compute validation softmax output 
                    val_output = network_predictions.eval(feed_dict={
                        network_input: X[test_index],
                        expected_output: Y[test_index],
                        gtp3_keep_prob: 1.0,
                        fc4_h_keep_prob: 1.0
                    })
                    
                    y_pred = np.array(val_output)
                    y_pred = y_pred[:,0]
                    y_true = (Y[test_index])[:,0]

                    aurocc = roc_auc_score(y_true, y_pred, average="weighted")
                    print y_pred, "\n", y_true
                    print aurocc
                    print validation_not_improved
                    if aurocc > latest_auc:
                        latest_auc = aurocc
                        validation_not_improved = 0
                    else:
                        validation_not_improved += 1
                    
                    if validation_not_improved > EARLY_STOPPING:
                        print "EARLY STOPPING"
                        print "--------------"
                        print "AUROCC :", latest_auc
                        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                        
                        # dump variables to plot later
                        dump_dict = {}
                        dump_dict["CNN"] = [fpr, tpr, aurocc, precision, recall]
                        with open('cnn_plot.pkl', 'wb') as f:
                            pickle.dump(dump_dict, f, pickle.HIGHEST_PROTOCOL)
                        break
            

            
            # Testing on test set
            
            output_dict = {}
            for i in range(test_set.dataset_size):

                file_name = test_set.file_names[i]

                if test_set.dropout_rate[i] > MAX_TEST_DROPOUT_TOLERANCE:
                    output_dict[file_name] = 0
                else:

                    test_input = [test_set.inputs[i]]
                    file_name = test_set.file_names[i]

                    feed_dict = {network_input: test_input, gtp3_keep_prob: 1.0,
                                fc4_h_keep_prob: 1.0}

                    output_prob = session.run(network_predictions, feed_dict=feed_dict)

                    # if preictal [1,0] we take the highest probability (e.g [0.7, 0.3]) that
                    # will be at index 0 and if interictal [0, 1], we take the lowest probability
                    # ([0.3 , 0.7]) that will also be at index 0 
                    if output_prob[0][0] < EPSILON:
                        output_prob[0][0] = 0

                    output_dict[file_name] = output_prob[0][0]
            
            self.write_submission_csv(output_dict, test_set.patient)

    
    def kernel_variable(self, shape):

        """ 
        It is the recommended initial kernels (or weights) for deep neural networks
        to break symetry, ensure consistant gradients backpropagation and avoid neurons saturation

        Generate initial kernel values following a truncated normal distribution 
        with a standard deviation of sqrt(2/Ni) rather than Xavier initialization
        following the recommandations of a recent paper: https://arxiv.org/pdf/1502.01852v1.pdf
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        (He and al. 2015)
        
        "This leads to a zero-mean Gaussian distribution whose standard deviation is sqrt(2/Ni)"

        shape is [kernel height, kernel width, number of input channels, number of output channels]

        Old initialization used : init_kernel = tf.truncated_normal(shape, stddev=0.01, mean=0.0, dtype=tf.float32)
        We can also use tf.random_uniform with [ -sqrt(2/Ni) ; sqrt(2/Ni) ] as boundary 
        """
        if len(shape) == 4:
            Ni = shape[0]*shape[1]*shape[2]
        else:
            Ni = shape[0]

        # init_kernel = tf.random_uniform(shape, minval=-math.sqrt(2.0/Ni), maxval=math.sqrt(2.0/Ni))
        init_kernel = tf.truncated_normal(shape, stddev=math.sqrt(2.0/Ni), mean=0.0, dtype=tf.float32)
        # init_kernel = tf.truncated_normal(shape, stddev=0.01, mean=0.0, dtype=tf.float32) #thesis (don't converge 0.5-0.5)

        return(tf.Variable(init_kernel))


    def bias_variable(self, shape):

        """ 
        Generate initial biases with a small positive value to avoid dead neurons

        shape is [number of output channels]
        """
        
        init_bias = tf.constant(0.1, shape=shape, dtype=tf.float32) # 0.1
        return(tf.Variable(init_bias)) 


    def write_submission_csv(self, output_dict, patient):

        """
        Write output dictionnary in a csv submission file
        """

        blank_submission_file_name = "../Submissions/blank/submission_"+patient+".csv"
        submission_file_name = "../Submissions/submission_"+patient+"model2.csv"
        df = pandas.read_csv(blank_submission_file_name)
        df["Class"] = df["Class"].astype("float")

        for line in range(len(df)):
            row_file_name = df.get_value(line, "File") 
            df = df.set_value(line, "Class", output_dict[row_file_name])

        df.to_csv(submission_file_name, index=False) 
        


for patient in range(1, 4):

    print "\nPatient:", patient
    print "----------\n"

    # Load training set and subsample
    trainingset = Trainingset("../Data/train_"+str(patient)+"/features/train_"+str(patient)+"_features.pgz",
                            "spectrogram", patient, use_validation = False, convNet = True)
    testset = Testset("../Data/test_"+str(patient)+"_new/features/test"+str(patient)+"_new_features.pgz",
                            "spectrogram")
    cnn_2 = CNN_2(trainingset, testset)
        
