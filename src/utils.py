
from __future__ import division

from matplotlib import pyplot as plt
import cPickle as pickle
import gzip
import numpy as np
import pandas


def save_zipped_pickle(obj, file_name, protocol = pickle.HIGHEST_PROTOCOL):
    """ Serialize a python object in a zipped pickle file """

    with gzip.GzipFile(file_name, 'wb') as handle:
        pickle.dump(obj, handle, protocol)


def load_zipped_pickle(file_name):
    """ Load a serialized python object from a zipped pickle file """

    with gzip.GzipFile(file_name, 'rb') as handle:
        obj = pickle.load(handle)

    return(obj)
    

def pretty_spectrogram(spectrogram):

    """
    Gent Master thesis spectrogram plot function
    """

    spectrogram = np.transpose(spectrogram, (2,1,0))

    ax = plt.gca()
    ax.set_yticks(range(0,6))
    ax.set_yticklabels([ 'delta', 'theta', 'alpha',
                        'beta', 'low-gamma', 'high-gamma'])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    ax.set_xticks(range(0,10))
    ax.set_xticklabels(range(0,10))
    plt.imshow(spectrogram[0, :, :], aspect='auto', origin='lower', interpolation='none')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel('Time, Epoch', fontsize=20)
    plt.show()


def pretty_signal(sig):

    """
    A simple signal plot using matplotlib
    """

    x = np.mgrid[0:len(sig)]
    y = sig[x]
    plt.plot(x,y)
    plt.xlabel("time", fontsize=20)
    plt.ylabel("amplitude", fontsize=20)
    plt.show()


def write_submission_csv(output_dict, blank_submission_file_path, destination_path):

    """
    Write output dictionnary in a csv submission file
    """
    
    print "Writing predictions to csv file.. ",

    df = pandas.read_csv(blank_submission_file_path)
    df["Class"] = df["Class"].astype("float")

    for line in range(len(df)):
        row_file_name = df.get_value(line, "File")
        try:
            df = df.set_value(line, "Class", output_dict[row_file_name])
        except:
            print "File name not in dictionary"
            continue

    df.to_csv(destination_path, index=False)

    print "done."

