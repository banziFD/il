import pickle
import numpy as np
import os 
import re

def cifar_unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
        return dict

def parse_meta(path):
    files = os.listdir(path)
    label_names = cifar_unpickle(path + "/batches.meta")[b'label_names']
    label_names = [i.decode('utf-8') for i in label_names]
    label_dict = dict((i, label_names[i]) for i in range(len(label_names)))
    return labels_names, labes_dict

def prepare_data():
    

def prepare_data_test():
