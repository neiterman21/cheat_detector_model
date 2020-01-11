#!/usr/local/bin python

import numpy as np # linear algebra
import cv2


class DataKeeper():
    def __init__(self, images , labels, label_names):

        self.image_paths = images
        self.labels = labels
        self.label_names = label_names
        self._batch_size = 16
        self._curent_index = 0


    def setBatchSize(self, batch_size):
        self._batch_size = batch_size

    def getNumOfBatches(self):
        return (int)(len(self.labels) / self._batch_size)

    def getNextBatch(self):
        if self._curent_index +  self._batch_size > len(self.labels):
            self._curent_index = 0
        data_image = self.image_paths[self._curent_index :  self._curent_index + self._batch_size]
        data_label = self.labels[self._curent_index :  self._curent_index + self._batch_size]
        lTr = np.array(data_label,dtype='float64')
        self._curent_index  += self._batch_size


        self._curent_index  += self._batch_size
        imTr = np.array(data_image, dtype='float32')

        imTr = np.ndarray.reshape(imTr,[imTr.shape[0],64,64])


        return imTr , lTr

