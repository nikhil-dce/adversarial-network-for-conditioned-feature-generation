__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

import tensorflow as tf
import numpy as np
import os

class DataHandler:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_train_data(self):

        train_class_file = os.path.join(self.data_dir, 'trainclasses.txt')
        train_label_file = os.path.join(self.data_dir, 'trainLabels')
        train_data_file = os.path.join(self.data_dir, 'trainData')
        train_attr_file = os.path.join(self.data_dir, 'trainAttributes')
        
        if not os.path.exists(train_class_file) or \
           not os.path.exists(train_label_file) or \
           not os.path.exists(train_data_file) or \
           not os.path.exists(train_attr_file):
            raise IOError("File cannot be read")

        with open(train_class_file, 'r') as f:
            data = f.readlines()

        data = [line.strip() for line in data]

        self.train_class_index, self.train_class_name = [], []

        for d in data:
            split_d = d.split('.')
            self.train_class_index.append(int(split_d[0]))
            self.train_class_name.append(split_d[1])

        with open(train_label_file) as f:
            self.train_label = np.load(f)
            
        with open(train_data_file) as f:
            self.train_data = np.load(f)

        with open(train_attr_file) as f:
            self.train_attr = np.load(f)

        self.train_size = self.train_data.shape[0]
        self.x_dim = self.train_data.shape[1]
        self.attr_dim = self.train_attr.shape[1]
        

    def next_train_batch(self, index, batch_size):
        start_index = index
        end_index = index+batch_size
        return self.train_data[start_index:end_index], self.train_attr[start_index:end_index]
