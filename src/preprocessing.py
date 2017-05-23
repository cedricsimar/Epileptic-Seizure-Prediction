# -*- coding: utf-8 -*- 

"""
 Extra features 
--------------
Mean-Squared Power Spectrum
msps = np.square(np.absolute(ft[0:nyq_index])) / samples_per_minut

Average Power 
ap = np.mean(msps)
print "Average Power : ", ap

Power Spectral Density
psd = msps / sampling_rate

Wavelet Coherence
Zero Crossing Rate
Fractal Dimension
Hurst Coefficient

"""


# imports
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import pandas
from pandas import DataFrame
import os
from shutil import copyfile
from matplotlib import pyplot as plt
import time

from utils import *

# Constants
PREICTAL = 1
INTERICTAL = 0

SAMPLING_FREQUENCY = 400

FREQUENCIES = np.array([0.4, 4, 8, 12, 30, 70, 180])    # 0.1
PSD_FREQ = np.array([[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]])
NUMBER_OF_BANDS = len(FREQUENCIES)-1
DEFAULT_NUMBER_OF_CHANNELS = 16
DEFAULT_NUM_EPOCHS = 10

DROPOUT = 0
SPECT_FLOOR = 0.0001


class Preprocessing:


    def preprocess_folder(self, folder_path):

        """
        Load every safe .mat file in folder, split the signal in no-dropout-epochs of fixed length
        (with overlap) and save the signals list, file name, hour, sequence and state in a new .mat file  
        """

        folder_path = folder_path if folder_path[-1] == '/' else folder_path + '/'
        self.is_testset = True if "test" in folder_path else False

        print "\nPreprocessing folder : " + folder_path + "\n"

        # create subfolder if it doesn't already exist
        sub_dir = folder_path + "30_sec_epochs/" 
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        else:
            print "30_sec_epochs folder already exists, no further action taken"
            return(0)

        nb_of_files = 0
        for f in os.listdir(folder_path): 
            if ".mat" in f: nb_of_files+=1
        
        if not self.is_testset:
            # extract the list of safe (not corrupted) files to train on
            safe_files_dic = self.parse_safe_file_csv()
        
        num_processed_file = 0
        
        for file_name in os.listdir(folder_path):

            if ".mat" not in file_name:
                continue
            
            num_processed_file += 1
            print "Preprocessing " + file_name + " ({0}/{1})".format(num_processed_file, nb_of_files)

            # if it is in the safe file list or part of the test set
            if self.is_testset or safe_files_dic[file_name][1]:

                # load file information in data_dict (get None if corrupted file)
                data_dict = self.load_file(folder_path + file_name)
                
                if data_dict != None:

                    new_file_struct = {}
                    new_file_struct['file_name'] = file_name

                    new_file_struct['data'] = self.split_data_in_epochs(data_dict['data'], 30, 15)
                    # [epoch][sample][channel]
                    
                    if not self.is_testset:
                        new_file_struct['state'] = data_dict['state']
                        new_file_struct['hour'] = data_dict['hour']
                        new_file_struct['sequence'] = int(data_dict['sequence'])
                                        
                    savemat(sub_dir + file_name, new_file_struct, appendmat=False, do_compression=True)
    

    def split_data_in_epochs(self, data, epoch_length_sec, stride_sec):

        """
        Split the signal in no-dropout-epochs of fixed length
        """
        
        sig = np.array(data, dtype=np.float32)  # [240000 x 16]

        sig_epochs = []
        samples_in_epoch = epoch_length_sec * SAMPLING_FREQUENCY
        stride_shift = stride_sec * SAMPLING_FREQUENCY

        # compute dropout indices (dropouts are at the same position across all channels)
        drop_indices_c0 = np.where(sig[:,0]==0)[0]
        drop_indices_c1 = np.where(sig[:,1]==0)[0]
        drop_indices = np.intersect1d(drop_indices_c0, drop_indices_c1)
        drop_indices = np.append(drop_indices, len(sig)) # add the index of the last element
        
        window_start = 0
        for window_end in drop_indices:
            
            epoch_start = window_start
            epoch_end = epoch_start + samples_in_epoch

            while(epoch_end < window_end):
                sig_epochs.append(sig[epoch_start:epoch_end, :])
                epoch_start += stride_shift
                epoch_end += stride_shift

            window_start = window_end + 1
        
        return(sig_epochs)  
    

    def parse_safe_file_csv(self):
        """ 
        Parse the safe file csv downloaded on Kaggle website to identity
        files that were not corrupted
        """

        safe_files = {} # dictionnary for faster access later (6k+ files to check)

        safe_file_name = "../Data/train_and_test_data_labels_safe.csv"
        df = pandas.read_csv(safe_file_name, ",")
        data = df.values
        
        # create a dictionnary with the filename as key and the label and safe check (1 if the file is safe
        # or 0 if the file is not safe) as values in an array

        for line in data:
            safe_files[line[0]] = [line[1], line[2]]

        return(safe_files)
    

    def copy_extra_preictals_from_testset(self, source_path, dest_path):

        """ RIP """
        
        safe_files_dic = self.parse_safe_file_csv()
        
        for file_name in os.listdir(source_path):
            try:
                if safe_files_dic[file_name][1]:
                    copyfile(source_path+file_name, dest_path+file_name)
            except:
                continue
        
    
    def extract_features_from_folder(self, folder_path):

        """ 
        Extract features from the 30-second windows of each files within the folder
        and save the features in the feature directory
        """

        folder_path = folder_path if folder_path[-1] == '/' else folder_path + '/'
        self.is_testset = True if "test" in folder_path else False

        if not os.path.exists(folder_path):
            print "Folder does not exist"
       
        print "Starting features extraction of folder " + folder_path + ":\n"

        nb_of_files = 0
        for f in os.listdir(folder_path): 
            if ".mat" in f: nb_of_files+=1
                
        num_processed_file = 0

        # loading features dictionary structured as
        # features_files_dict[file_name][feature_name][epoch][..]

        features_path = folder_path + "../" + "features/"
        features_file_name = folder_path.split("/")[-3] + "_features.pgz"

        print "Loading features dictionary.. "
    
        if os.path.exists(features_path + features_file_name):
            # if the feature file already exists
            features_files_dict = load_zipped_pickle(features_path+features_file_name)
            # features_files_dict = loadmat(features_path + features_file_name)
            print "Features file has been found and loaded.\n"
        else:
            features_files_dict = {}
            try:
                os.makedirs(features_path)
                print "Features folder and file not found. Folder created..\n"
            except:
                print "Features folder found but file not found. Creating new dictionary..\n"

        for file_name in os.listdir(folder_path):

            # extract features from file
            print "Extracting features from file " + file_name + " (" + str(num_processed_file+1) + "/" + str(nb_of_files) + ")"
            
            try:
                file_data = loadmat(folder_path + file_name)
            except:
                print "Could not open corrupted file : " + file_path
                continue
                        
            signals_data = file_data['data']    # [30_sec_window][sample][channel]

            features_files_dict = self.extract_features_from_signals(features_files_dict, file_name, signals_data)
            
            # add relevant training set information if applicable
            if not self.is_testset:
                features_files_dict[file_name]['state'] = int(file_data['state'])
                features_files_dict[file_name]['hour'] = int(file_data['hour'])
            
            num_processed_file+=1

        
        print "\nFeatures extraction done.\n"
        
        # save features dictionary in a .pgz file
        
        print "Saving features dictionary to file", features_file_name + "..",
        save_zipped_pickle(features_files_dict, features_path+features_file_name)

        print " done.\n"


    def extract_features_from_signals(self, features_files_dict, file_name, signals_data):

        """
        Compute and return a features dictionary with the following structure:
            dict['spectrogram'] = [window][spectrogram (96x10)]
            dict['flattened_spectrogram] = [window][flattened spectrogram (1x96)]
            
        To be implemented:

            # Spectral Coherence ou Alpha / Beta .. Coherences
            # Wavelet Coherence
            # Short-Term Energy
            # Zero Crossing Rate
            # Fractal Dimension
            # Hurst Coefficient
        """ 

        implemented_features_list = ['spectrogram', 'flattened_spectrogram', 'power_spectral_density']

        # check if there already is some features with file_name
        if file_name not in features_files_dict:
            features_files_dict[file_name] = {}
        
        # check if some features are not already present in the dict
        for feature in implemented_features_list:
            if feature not in features_files_dict[file_name]:
                features_files_dict[file_name][feature] = []
        
        number_of_channels = 16                         
        nyquist_frequency = SAMPLING_FREQUENCY / 2.              
        number_of_windows = len(signals_data)
        seconds_in_window = 30
        seconds_in_epoch = seconds_in_window / 10
    
        # compute the features for each window indepedently
        # windowed signal is not modified

        for windowed_signal in signals_data:

            # Spectrograms
            # ------------

            # apply a low pass filter
            lp_filtered_signal = self.low_pass_filter(windowed_signal, 0.4, 180, nyquist_frequency)

            # compute 10-epochs spectrograms
            if len(features_files_dict[file_name]['spectrogram']) < len(signals_data):

                amp_spect = self.compute_spectrogram(lp_filtered_signal, seconds_in_window, SAMPLING_FREQUENCY,
                                                    NUMBER_OF_BANDS, seconds_in_epoch, seconds_in_epoch)
                amp_spect = self.to_96x10_spectrogram(amp_spect)

                features_files_dict[file_name]['spectrogram'].append(amp_spect)

            # compute 1-epoch flatten spectrogram
            if len(features_files_dict[file_name]['flattened_spectrogram']) < len(signals_data):

                flat_spect = self.compute_spectrogram(lp_filtered_signal, seconds_in_window, SAMPLING_FREQUENCY,
                                                    NUMBER_OF_BANDS, seconds_in_window, seconds_in_window)
                flat_spect = np.ndarray.flatten(flat_spect)

                features_files_dict[file_name]['flattened_spectrogram'].append(flat_spect)
            
            # compute power spectral density
            if len(features_files_dict[file_name]['power_spectral_density']) < len(signals_data):
                
                psd = self.compute_power_spectral_density(windowed_signal.T)
                psd = np.ndarray.flatten(psd) # 6x16 to 1x96
                features_files_dict[file_name]['power_spectral_density'].append(psd)

        #}
        return(features_files_dict)
    

    def compute_power_spectral_density(self, windowed_signal):

        # Windowed signal of shape [channel][sample] [16 x 12000]
        ret = []
        
        # Welch parameters
        sliding_window = 512
        overlap = 0.25
        n_overlap = int(sliding_window * overlap)
        
        # compute psd using Welch method
        freqs, power = signal.welch(windowed_signal, fs=SAMPLING_FREQUENCY,
                                    nperseg=sliding_window, noverlap=n_overlap)
        
        for psd_freq in PSD_FREQ:
            tmp = (freqs >= psd_freq[0]) & (freqs < psd_freq[1])
            ret.append(power[:,tmp].mean(1))
        
        return(np.log(np.array(ret) / np.sum(ret, axis=0)))

        
    def to_96x10_spectrogram(self, amp_spect):

        """
        Reshape the data to (96,10) ready for the
        first convolution and the desired output to the shape of the network output

        The reshape is done to have a 2d input as follow: 

                   | frequency band 6
        channel 1  | frequency band 5
                   | ...                          ...
                   | frequency band 1
        ...        |------------------   ... 
                   | frequency band 6
        channel 16 | frequency band 5
                   | ...                          ...
                   | frequency band 1

                        minute 1         ...    minute 10


        Another reshape to be tested is to group lines by frequecy bands rather than by channels
        to try to facilitate the identification of channel coherence for specific frequency band.


        frequency band 1 (channel 1)
        frequency band 1 (channel 2)
        ...                                                   ...
        frequency band 1 (channel 16)
        
        ...                                 ...

        frequency band 6 (channel 1)
        frequency band 6 (channel 2)
        ...                                                   ...
        frequency band 6 (channel 16)

                minute 1                     ...            minute 10

        This reshape can be easily done with np.reshape(10,96) -> np.reshape(96,10)
        """

        new_x = [[None for j in range(DEFAULT_NUM_EPOCHS)] for i in range(DEFAULT_NUMBER_OF_CHANNELS*NUMBER_OF_BANDS)]
        line = 0

        while(line < (DEFAULT_NUMBER_OF_CHANNELS*NUMBER_OF_BANDS)):
            for channel in range(DEFAULT_NUMBER_OF_CHANNELS):
                for band in range(NUMBER_OF_BANDS):
                    for epoch in range(DEFAULT_NUM_EPOCHS):
                            new_x[line][epoch] = amp_spect[epoch][band][channel]
                    line+=1
        
        return(np.array(new_x))
    

    def low_pass_filter(self, sig, low_cut, high_cut, nyq):

        """ 
        Apply a low pass filter to the data to take the relevant frequencies
        and to smooth the drop-out regions

        No resample necessary because all files have the same sampling frequency
        """

        b, a = signal.butter(5, np.array([low_cut, high_cut]) / nyq, btype='band')
        sig_filt = signal.lfilter(b, a, sig, axis=0)
       
        return(np.float32(sig_filt))
    

    def hanning(self, sig):

        """
        Apply a Hanning window to the signal and return it
        """ 

        han_window = signal.hann(len(sig))
        return(sig*han_window)
    

    def compute_spectrogram(self, sig, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec):
        
        n_channels = 16
        n_timesteps = int((data_length_sec - win_length_sec) / stride_sec + 1)
        n_fbins = nfreq_bands

        sig = np.transpose(sig)

        sig2 = np.zeros((n_channels, n_fbins, n_timesteps))
        for i in range(n_channels):
            sigc = np.zeros((n_fbins, n_timesteps))
            for frame_num, w in enumerate(range(0, int(data_length_sec - win_length_sec + 1), stride_sec)):

                sigw = sig[i, w * sampling_frequency: (w + win_length_sec) * sampling_frequency]
                sigw = self.hanning(sigw)
                fft = self.log10(np.absolute(np.fft.rfft(sigw)))
                fft_freq = np.fft.rfftfreq(n=sigw.shape[-1], d=1.0 / sampling_frequency)
                sigc[:nfreq_bands, frame_num] = self.group_into_bands(fft, fft_freq, nfreq_bands)

            sig2[i, :, :] = sigc

        return np.transpose(sig2, axes=(2,1,0))
    

    def log10(self, array):

        """ 
        Custom log10 method to avoid -inf due to log10(0)

        It produces a divide by zero warning because np.where creates both arrays (the one for
        the positive and negative case) and select each element afterwards. 
        But the warning can now safely be ingored
        """

        return(np.where(array > DROPOUT, np.log10(array), SPECT_FLOOR))


    def group_into_bands(self, fft, fft_freq, nfreq_bands):

        """ 
        Group the fft result by frequency bands and take the mean
        of the fft values within each band

        Return a list of the frequency bands' means (except the first element
        which is the frequency band 0 - 0.1Hz)
        """

        freq_bands = np.digitize(fft_freq, FREQUENCIES)
        df = DataFrame({'fft': fft, 'band': freq_bands})
        df = df.groupby('band').mean()
        return df.fft[1:-1]

    
    def load_file(self, file_path):
       
        """
        Extract information from .mat file and structure it in data_dict
        A file represents 10 minutes of iEEG sample
        
        loadmat comes from scipy.io.loadmat and returns a dictionary with variable 
        names as keys, and loaded matrices as values
        
        strucutre of file_data :
          dataStruct
          __version__
          __header__
          __globals__
        
        structure of ['dataStruct'] : 
        ('data', 'iEEGsamplingRate', 'nSamplesSegment', 'channelIndices', 'sequence')

        data : the number of data samples representing 1 second of EEG data [240000 x 16]
        iEEGsamplingRate = 400.0 Hz (the number of data samples representing 1 second of EEG data. )
        nSamplesSegment = 240 000 (number of values for 10 minutes : 400*60*10)
        channelIndices = [1, 2, ..., 16] an array of the electrode indexes corresponding 
                                         to the columns in the data field. 
        sequence = the index of the data segment within the one hour series of clips. 
                   For example, 1_12_1.mat has a sequence number of 6, and represents 
                   the iEEG data from 50 to 60 minutes into the preictal data. This field 
                   only appears in training data. 
        """

        try:
            file_data = loadmat(file_path)
        except:
            print "/!\ Could not open corrupted file : " + file_path
            return(None)
        
        # structuring information in file_data['dataStruct'] in a dictionary
        data_dict = {name: file_data['dataStruct'][name][0,0] for name in file_data['dataStruct'].dtype.names}

        # if it's training set data, adding information on the state 
        # and hour of recording (for cross validation)
        
        if not self.is_testset:
            
            # state is either preictal or interictal
            tmp = file_path.split('/')[-1].split('_')
            
            data_dict['state'] = int(tmp[2].split('.')[0])
            data_dict['hour'] = (int(tmp[1])-1) / 6          # for grouped cross validation

            
        return(data_dict)               



# Preprocessing 
# -------------

preprocessing = Preprocessing()

# Preprocess training and test folders
for patient in range(1, 4):
    preprocessing.preprocess_folder("../Data/train_"+str(patient)+"/")
    preprocessing.preprocess_folder("../Data/test_"+str(patient)+"_new/")

# Extract features from training and test 30-sec epochs
for patient in range(1, 4):
    preprocessing.extract_features_from_folder("../Data/train_"+str(patient)+"/30_sec_epochs/")
    preprocessing.extract_features_from_folder("../Data/test_"+str(patient)+"_new/30_sec_epochs/")
    
