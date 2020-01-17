#!/usr/bin/env python

import sys
from DDetector import *
from sklearn.model_selection import train_test_split
sys.path.append('utils/')
from itertools import zip_longest
import fsdd

DeceptionDB_path = "data/DeceptionDB/"
DeceptionDB_csv_path = "data/DeceptionDB/description.csv"

test_db_data_path = "data/num_rec_data/"
test_db_spectro_data_path = test_db_data_path + 'spectro/'

#contains spectrograms from wav files. same file name different ending wav <--> png
spectrogram_dir = "data/spectrograms/"
batch_size = 64
data_chunks = 1

import signal
import sys

def grouper(n, iterable, padvalue=None):
    chunks = lambda l, n: [l[x: x+n] for x in range(0, len(l), n)]
    return chunks(iterable, int(len(iterable)/n))

def sigint_handler(signal, frame):
    print("Accuracy history: ", RNN.accuracy_history)
    sys.exit(0)

def confData(data_audio_train, data_audio_test , data_label_train , data_label_test):
    print("train data len: " + str(len(data_audio_train)))
    print("test data len: " + str(len(data_audio_test)))
    train_data  = Datakeeper.DataKeeper(data_audio_train,data_label_train, ["Lie" , "True"] )
    train_data.setBatchSize(batch_size)
    test_data   = Datakeeper.DataKeeper(data_audio_test,data_label_test, ["Lie" , "True"] )
    test_data.setBatchSize(batch_size)
    return train_data , test_data

def train_test_chunk_split(audio_list , label_list , test_chunk):
    data_audio_train    = []
    data_audio_test     = []
    data_label_train    = []
    data_label_test     = []
    for i in range(data_chunks):
        if i == test_chunk:
            data_label_test = label_list[i]
            data_audio_test = audio_list[i]
            continue

        data_label_train.extend(label_list[i])
        data_audio_train.extend(audio_list[i])
    return data_audio_train, data_audio_test , data_label_train , data_label_test

def run_on_chunk(network , audio_lists , label_lists , chunk):
    print("Running on chunk number" , chunk , " as test set")

   # data_audio_train, data_audio_test , data_label_train , data_label_test = train_test_chunk_split(audio_lists , label_lists , chunk)
    data_audio_train, data_audio_test , data_label_train , data_label_test = train_test_split(audio_lists , label_lists , test_size=0.20)

    train_data , test_data = confData(data_audio_train, data_audio_test , data_label_train , data_label_test)
    return RNN.run_RNN(network[0] ,network[1] , network[2] ,network[3] , train_data , test_data ,400)


def main():
    print("running DDetector")
    db = fsdd.FSDD(spectrogram_dir)

    #get lists of audio samples and labels as numpy array
    audio_list , label_list = db.get_spectrograms(spectrogram_dir)

    train_op ,accuracy , loss_op, pred = RNN.configure_RNN()
    train_op ,accuracy , loss_op, pred = cnn_example.configure_CNN()

    rnn = (train_op ,accuracy , loss_op, pred)
    label_lists = grouper(data_chunks,label_list)
    audio_lists = grouper(data_chunks,audio_list)

    #splint data randomly for train and test
    #data_audio_train, data_audio_test , data_label_train , data_label_test = train_test_split(audio_list , label_list , test_size=0.20)
    pref = []
    for i in range(data_chunks):
        pref.append(run_on_chunk(rnn , audio_list , label_list , i))
    printPref(pref)



def printPref(pref):
    print("printing total preformans")
    avg_acc = 0
    full_mat = [[0,0],[0,0]]
    testchunk = 0
    for p in pref:
        print("test chunk: " , testchunk)
        testchunk += 1
        print("Acuracy = " , p[0])
        avg_acc += p[0]
        m = p[1]
        print (m[0][0] , m[0][1])
        print (m[1][0] , m[1][1])

        full_mat[0][0] += m[0][0]
        full_mat[0][1] += m[0][1]
        full_mat[1][0] += m[1][0]
        full_mat[1][1] += m[1][1]

    print("Avg Acuracy = " , avg_acc/data_chunks)
    print("full mat: ")
    print (full_mat[0][0] , full_mat[0][1])
    print (full_mat[1][0] , full_mat[1][1])


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    main()

