from __future__ import print_function
import os
from collections import defaultdict
import scipy.io.wavfile
import scipy.ndimage
import numpy as np


class FSDD_:
    """Summary

    Attributes:
        file_paths (TYPE): Description
        recording_paths (TYPE): Description
    """

    def __init__(self, data_dir):
        """Initializes the FSDD data helper which is used for fetching FSDD data.

        :param data_dir: The directory where the audiodata is located.
        :return: None

        Args:
            data_dir (TYPE): Description
        """

        # A dict containing lists of file paths, where keys are the label and vals.
        self.recording_paths = defaultdict(list)
        file_paths = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        self.file_paths = file_paths

        for digit in range(0, 10):
            # fetch all the file paths that start with this digit
            digit_paths = [os.path.join(data_dir, f) for f in file_paths if f[0] == str(digit)]
            self.recording_paths[digit] = digit_paths

    @staticmethod
    def get_lable(file_name):
        return file_name[0]

    @classmethod
    def get_spectrograms(cls, data_dir=None):
        """

        Args:
            data_dir (string): Path to the directory containing the spectrograms.

        Returns:
            (spectrograms, labels): a tuple of containing lists of spectrograms images(as numpy arrays) and their corresponding labels as strings
        """
        spectrograms = []
        labels = []

        if data_dir is None:
            data_dir = os.path.dirname(__file__) + '/../spectrograms'

        file_paths = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and '.png' in f]

        if len(file_paths) == 0:
            raise Exception('There are no files in the spectrogram directory. Make sure to run the spectrogram.py before calling this function.')

        for file_name in file_paths:
            label = cls.get_lable(file_name)
            if label is None:
                continue
            spectrogram = scipy.ndimage.imread(data_dir + '/' + file_name, flatten=True).flatten()
            spectrograms.append(spectrogram)
            labels.append(label)

        print("total num of entris is: " + str(FSDD.num_of_entries))
        return spectrograms, np.array(labels)

class FSDD(FSDD_):
    import read_data
    csv_file = "data/DeceptionDB/description.csv"
    data_labels = read_data.parst_data_labels(csv_file)
    del read_data
    num_of_entries = 0
    def __init__(self, data_dir ):
        super(FSDD, self).__init__(data_dir)

    #overide
    @staticmethod
    def get_lable(file_name):
        print(file_name)
        entry = next((x for x in FSDD.data_labels if x["filename"].replace("wav","png") == file_name) , None)
        if entry is None:
            return
        FSDD.num_of_entries = FSDD.num_of_entries + 1
        if entry["isliestatment"] == "True":    # [is lie , is true]
            return [0,1]
        else:
            return [1,0]

