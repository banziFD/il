import pickle
import glob
import numpy as np
from random import shuffle

# create an annotation based dataset for later experiment

class AnnTool:
    def __init__(self, path):
        self.path = path + '*.txt';
        files = glob.glob(self.path)
        # format:[[label, bbox, imagename], ...
        self.ann = list()
        for fname in files:
            f = open(fname)
            lines = f.readlines()
            for line in lines:
                line = line.split()
                image = line[0]
                label = line[1]
                label = int(label)
                bbox = line[2:]
                bbox = [float(i) for i in bbox]
                current_ann = [label, bbox, image]
                self.ann.append(current_ann)

    def get_annotations(self, label = -1):
        if(label == -1):
            return self.ann
        else:
            label_ann = list()
            for item in self.ann:
                if(item[0] == label):
                    ann.append(item)
            return label_ann
    
    @staticmethod
    def save(content, filename):
        try:
            pickle.dump(content, open(filename, 'wb'))
        except ValueError as e:
            print(e)
    @staticmethod 
    def load(filename):
        try:
            data = pickle.load(open(filename, 'rb'))
            return data
        except ValueError as e:
            print(e)
            return None
    
    @staticmethod
    def label2numpy(label):
        pass
    
    @staticmethod
    def bbox2numpy(bbox):
        pass