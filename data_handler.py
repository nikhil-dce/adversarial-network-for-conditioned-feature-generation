__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

import tensorflow as tf
import numpy as np
import os

class DataHandler:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.test_data_loaded = False

    def load_data(self):

        train_class_file = os.path.join(self.data_dir, 'trainclasses_ps.txt')
        train_label_file = os.path.join(self.data_dir, 'trainLabels')
        train_data_file = os.path.join(self.data_dir, 'trainData')
        train_attr_file = os.path.join(self.data_dir, 'trainAttributes')

        test_class_file = os.path.join(self.data_dir, 'testclasses_ps.txt')
        test_label_file = os.path.join(self.data_dir, 'testLabels')
        test_data_file = os.path.join(self.data_dir, 'testData')
        all_attr_file = os.path.join(self.data_dir, 'dataAttributes')

        all_class_file = os.path.join(self.data_dir, 'allclasses.txt')
        
        if not os.path.exists(train_class_file) or \
           not os.path.exists(train_label_file) or \
           not os.path.exists(train_data_file) or \
           not os.path.exists(train_attr_file) or \
           not os.path.exists(test_class_file) or \
           not os.path.exists(test_label_file) or \
           not os.path.exists(test_data_file) or \
           not os.path.exists(all_class_file) or \
           not os.path.exists(all_attr_file):
            raise IOError("File cannot be read")

        self.all_classes = {}
        with open(all_class_file, 'r') as f:
            data = f.readlines()
            for idx, cl in enumerate(data):
	        self.all_classes[cl.split()[0]] = idx

        with open(test_class_file, 'r') as f:
            data = f.readlines()
            self.test_classes = [self.all_classes[x.split()[0]] for x in data]

        with open(train_class_file, 'r') as f:
            data = f.readlines()
            self.train_classes = [self.all_classes[x.split()[0]] for x in data]

        with open(all_attr_file) as f:
            self.all_attr = np.load(f)

        # Train files load
        self.load_train_data(train_class_file, train_label_file, train_data_file, train_attr_file)
        self.load_test_data(test_class_file, test_label_file, test_data_file, all_attr_file)
        

    def preprocess_data(self):

        print 'Preprocess'
        
        # Do everything here so i can remove this function if i want to
        # Preprocess c and x
        self.epsilon = 1e-6
        self.attr_mean = np.mean(self.all_attr, axis=0, keepdims=True) # Note all_attr (Seen and Unseen classes) available at training time
        self.attr_std = np.std(self.all_attr, axis=0, keepdims=True)

        print self.all_attr.shape
        print self.attr_mean.shape
        print self.attr_std.shape

        self.train_attr = np.divide(self.train_attr - self.attr_mean,  (self.attr_std + self.epsilon))
        self.all_attr = np.divide(self.all_attr - self.attr_mean,  (self.attr_std + self.epsilon))
        self.test_attr = np.divide(self.test_attr - self.attr_mean,  (self.attr_std + self.epsilon))
        
        self.train_data_mean = np.mean(self.train_data, axis=0, keepdims=True)
        self.train_data_std = np.std(self.train_data, axis=0, keepdims=True)

        self.train_data = np.divide(self.train_data - self.train_data_mean,  (self.train_data_std + self.epsilon))
        # Here only preprocessing the test data
        # Note: Test data has not been used for preprocessing (calculation of mean or std)
        self.test_data = np.divide(self.test_data - self.train_data_mean,  (self.train_data_std + self.epsilon))
        
    def load_train_data(self, train_class_file, train_label_file, train_data_file, train_attr_file):
        
        with open(train_label_file) as f:
            self.train_label = np.load(f)
            
        with open(train_data_file) as f:
            self.train_data = np.load(f)

        with open(train_attr_file) as f:
            self.train_attr = np.load(f)

        self.train_size = self.train_data.shape[0]
        self.x_dim = self.train_data.shape[1]
        self.attr_dim = self.train_attr.shape[1]

        print 'Training Data: ' + str(self.train_data.shape)
        print 'Training Attr: ' + str(self.train_attr.shape)
        
    def load_test_data(self, test_class_file, test_label_file, test_data_file, all_attr_file):

        with open(test_label_file) as f:
            self.test_label = np.load(f)
            
        with open(test_data_file) as f:
            self.test_data = np.load(f)

        self.test_attr = self.all_attr[self.test_classes]
        self.test_size = self.test_data.shape[0]
        
        print 'Testing Data: ' + str(self.test_data.shape)
        print 'Testing Attr: ' + str(self.test_attr.shape)
        print 'Testing Classes' + str(len(self.test_classes))
        print 'Testin Labels' + str(self.test_label.shape)

    def next_train_batch(self, index, batch_size):
        start_index = index
        end_index = index+batch_size
        return self.train_data[start_index:end_index], self.train_attr[start_index:end_index]

    def next_test_batch(self, index, batch_size):
        start_index = index
        end_index = index+batch_size
        return self.test_data[start_index:end_index], self.test_attr[start_index:end_index]
