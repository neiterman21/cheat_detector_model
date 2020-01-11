#!/usr/bin/env python

import sys
from DDetector import *
from sklearn.model_selection import train_test_split
sys.path.append('utils/')
import numpy as np
import fsdd

DeceptionDB_path = "data/DeceptionDB/"
DeceptionDB_csv_path = "data/DeceptionDB/description.csv"

test_db_data_path = "data/num_rec_data/"
test_db_spectro_data_path = test_db_data_path + 'spectro/'

#contains spectrograms from wav files. same file name different ending wav <--> png
spectrogram_dir = "data/spectrograms/"
batch_size = 64

import signal
import sys



def sigint_handler(signal, frame):
    print("Accuracy history: ", RNN.accuracy_history)
    sys.exit(0)

def main():
    print("running DDetector")
    db = fsdd.FSDD(spectrogram_dir)

    #get lists of audio samples and labels as numpy array
    audio_list , label_list = db.get_spectrograms(spectrogram_dir)


    #splint data randomly for train and test
    data_audio_train, data_audio_test , data_label_train , data_label_test = train_test_split(audio_list , label_list , test_size=0.15)

    print("train data len: " + str(len(data_audio_train)))
    print("test data len: " + str(len(data_audio_test)))
    train_data  = Datakeeper.DataKeeper(data_audio_train,data_label_train, ["Lie" , "True"] )
    train_data.setBatchSize(batch_size)
    test_data   = Datakeeper.DataKeeper(data_audio_test,data_label_test, ["Lie" , "True"] )
    test_data.setBatchSize(batch_size)
    train_op ,accuracy , loss_op, pred = RNN.configure_RNN()


    RNN.run_RNN(train_op ,accuracy , loss_op , train_data , test_data , pred,10000)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    main()

