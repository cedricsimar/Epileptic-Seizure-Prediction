from __future__ import division

import numpy as np
import os
from matplotlib import pyplot as plt
import random
from copy import copy
from utils import load_zipped_pickle

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

VALIDATION_PERCENTAGE = 0.20
MAX_DROPOUT_RATE = 0.5


class Trainingset:

    def __init__(self, path_to_trainingset, selected_features, patient, use_validation = False, convNet = False):

        self.patient_number = patient
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.trainingset_mean = None
        self.trainingset_var = None

        self.validationset = None
        self.validationlabels = None
        self.selected_features = selected_features

        # load training samples, labels and groups/hours from file
        self.inputs, self.labels, self.groups = self.load_dataset(path_to_trainingset)
        assert(len(self.inputs) == len(self.labels))
        self.dataset_size = len(self.inputs)

        # if validation, split the training set before normalizing
        if use_validation: 
            self.split_training_validation(VALIDATION_PERCENTAGE)
        else:
            self.undersampling()

        # convert training and validation labels into one-hot
        if convNet:
            self.labels = self.to_one_hot(self.labels)



    def undersampling(self):

        """ 
        To counter the highly imbalanced dataset (e.g. 1/16 in train_3), we under-sample
        the majority class (interictal) after the data augmentation during preprocessing.  

        The proportions of preictals / total samples per patient are the following: 
          - patient 1 : 35%
          - patient 2 : 11%
          - patient 3 : 12%
        
        It was defined empirically
        """

        # reduce the number of interictal files to
        n_interictals = [6600, 35485, 39676]
        # in order to keep the proportion

        # pick indices of all preictal / interical samples
        i_preic = np.where(self.labels == 1)[0]
        i_inter = np.where(self.labels == 0)[0]

        # shuffle the interictal indices and only pick the number of preictal segments
        # in order to have a (reduced) balanced dataset
        np.random.shuffle(i_inter)  # in place
        i_inter = i_inter[:n_interictals[self.patient_number-1]]

        # concatenate the the indices arrays and select the desired inputs and labels
        selected = np.concatenate((i_preic, i_inter))
        np.random.shuffle(selected)

        self.inputs = self.inputs[selected]
        self.labels = self.labels[selected]
        self.groups = self.groups[selected]

        self.dataset_size = len(self.inputs)


    def normalize_inputs(self):

        """
        In the case of the training set, we calculate the mean and variance over
        the whole dataset
        
        In the case of the test set, we use the mean and variance of the training set
        for the test set normalization

        DEPRECATED

        """

        # concatenate all input into an array
        all_inputs = self.inputs[0]
        for i in range(1, self.dataset_size):
            all_inputs = np.concatenate((all_inputs, self.inputs[i]), axis=0)
        
        # calculate the mean and standard deviation
        mean = np.mean(all_inputs, 0)
        std = np.std(all_inputs, 0)

        # save them for test or validation sets
        self.trainingset_mean = mean
        self.trainingset_var = std

        # normalize inputs (inputs = (inputs - mean)/std)
        for i in range(self.dataset_size):
            self.inputs[i] = (self.inputs[i] - mean) / std
        

    def next_batch(self, batch_size = 10):

        """ return the next batch of size batch_size from the training set """
        
        assert(batch_size <= self.dataset_size)

        start_index = self.index_in_epoch
        self.index_in_epoch += batch_size

        if(self.index_in_epoch > self.dataset_size):
            # epoch completed
            self.epochs_completed += 1
            
            # shuffle the data
            permutations = np.arange(self.dataset_size)
            np.random.shuffle(permutations)
            self.inputs = self.inputs[permutations]
            self.labels = self.labels[permutations]

            # start new epoch
            start_index = 0
            self.index_in_epoch = batch_size
        
        end_index = self.index_in_epoch
        
        return( self.inputs[start_index:end_index], self.labels[start_index:end_index] )
                    

    def load_dataset(self, path_to_dataset):

        """ 
        Load the features .pgz file and parse it into the class variables
        """ 
                    
        inputs = []
        labels = []

        # creating groups for stratified validation (all 10-minute file from a specific
        # hour can only be or in training set or in validation set)
        groups = []

        print "Loading dataset using features " + str(self.selected_features) + "..",

        # [file_name][feature_name][epoch][..] or
        # [file_name][state/hour]
        dataset_dict = load_zipped_pickle(path_to_dataset) 
        
        for file in dataset_dict:
            for input in dataset_dict[file][self.selected_features]:
                inputs.append(input)
                labels.append(dataset_dict[file]['state'])
                groups.append(dataset_dict[file]['hour'])
                        
        print(" done.")
        
        return(np.array(inputs), np.array(labels), np.array(groups))
        

    def to_one_hot(self, array):

        """
        Convert preictal / interictal output (1/0) into one-hot representation for
        classification (respectively [1,0], [0,1])
        """

        array = list(array)
        for i in range(len(array)):
            array[i] = PREICTAL_OUTPUT if array[i] == PREICTAL else INTERICTAL_OUTPUT
        
        return(np.array(array))


    def split_training_validation(self, percentage):

        """
        Group stratified validation set: Every hour (group of 6 x 10 minutes file) of the training set 
        is placed either in the training set, or in the validation set to avoid over-optimistic 
        validation score due to the fact than 10-minute samples that belong to the same hour 
        segment are easier to recognize than 10-minute samples from another hour segment.

        The validation is stratified, it keeps (at its best) the same proportion of preictal and interictal
        samples in the training and validation sets.
        """ 

        print "Splitting dataset in to training and validation sets.. ",
        
        number_of_inputs = len(self.inputs)
        cut_index = int(number_of_inputs * percentage)

        # shuffle
        permutations = np.arange(number_of_inputs)
        np.random.shuffle(permutations)
        self.inputs = self.inputs[permutations]
        self.labels = self.labels[permutations]
        
        # self.validationset = self.inputs[0:cut_index]
        # self.validationlabels = self.labels[0:cut_index]

        # self.inputs = self.inputs[cut_index:]
        # self.labels = self.labels[cut_index:]

        unique_groups = np.unique(self.groups)
        nb_of_groups = len(unique_groups)

        # shuffle the group numbers
        permutations = np.arange(nb_of_groups)
        np.random.shuffle(permutations)
        unique_groups = unique_groups[permutations]

        nb_of_preictals = len(np.where(self.labels == PREICTAL)[0])
        nb_of_interictals = len(self.labels) - nb_of_preictals

        max_preictals_in_validation = int(nb_of_preictals * percentage) + 1
        max_interictals_in_validation = int(nb_of_interictals * percentage) + 1 

        nb_preictals_in_validation = 0
        nb_interictals_in_validation = 0

        validation_indexes = np.array([], dtype=np.int64)
        
        # each group can have preictal, interictal samples or both
        for group in unique_groups:
            # get all samples from that group
            group_samples_index = np.where(self.groups == group)[0]

            group_preictal_indexes = []
            group_interictal_indexes = []

            # sort the group sample indexes into preictal or interictal
            for sample_index in group_samples_index:
                if self.labels[sample_index] == PREICTAL:
                    group_preictal_indexes.append(sample_index)
                else:
                    group_interictal_indexes.append(sample_index)
            
            if nb_interictals_in_validation < max_interictals_in_validation:
                validation_indexes = np.concatenate((validation_indexes, group_interictal_indexes))
                nb_interictals_in_validation += len(group_interictal_indexes)
            
            if nb_preictals_in_validation < max_preictals_in_validation:
                validation_indexes = np.concatenate((validation_indexes, group_preictal_indexes))
                nb_preictals_in_validation += len(group_preictal_indexes)
            
            if nb_interictals_in_validation >= max_interictals_in_validation and nb_preictals_in_validation >= max_preictals_in_validation:
                break # my heart
        
        # update training and validation sets
        validation_indexes = validation_indexes.astype(np.int64)
        self.validationset = self.inputs[validation_indexes]
        self.validationlabels = self.labels[validation_indexes]

        self.inputs = np.delete(self.inputs, validation_indexes, axis=0)
        self.labels = np.delete(self.labels, validation_indexes, axis=0)
        
        # update dataset size
        self.dataset_size = len(self.inputs)

        print("done.")



class Testset:

    def __init__(self, path_to_testset, selected_features):

        self.selected_features = selected_features

        # load training samples, labels and groups/hours from file
        self.inputs, self.file_names = self.load_dataset(path_to_testset)
        self.dataset_size = len(self.inputs)


    def load_dataset(self, path_to_dataset):

        """ 
        Load the features .pgz file and parse it into the class variables
        """ 
                    
        inputs = []
        file_names = []
        
        print "Loading testset using features " + str(self.selected_features) + "..",

        # [file_name][feature_name][epoch][..] and
        # [file_name][state/hour]
        dataset_dict = load_zipped_pickle(path_to_dataset) 
        
        for file in dataset_dict:
            inputs.append(dataset_dict[file][self.selected_features])
            file_names.append(file)
                        
        print(" done.")
        
        return(np.array(inputs), np.array(file_names))
        

    
