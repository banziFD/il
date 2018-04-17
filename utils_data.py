import pickle
import numpy as np
import glob
import torch.utils.data
import torch
import torchvision.transforms as transforms

def cifar_unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
        return dict

def parse_meta(path):
    label_names = cifar_unpickle(path + "/batches.meta")[b'label_names']
    label_names = [i.decode('utf-8') for i in label_names]
    label_dict = dict((i, label_names[i]) for i in range(len(label_names)))
    return labels_names, labes_dict

def prepare_format(labels, images):
    labels = np.array(labels)
    images = np.array(images)
    images = images.reshape(images.shape[0], 32 * 32, 3)
    images = images.reshape(images.shape[0], 3 , 32, 32).transpose([0, 2, 3, 1])
    return labels, images

def prepare_files(dataset_path, work_path, mixing):
    # prepare image/label files according to mixing, who
    # will be load into model using dataloader
    # mixing format: [(class1, class2... in group1),
    # (class1, class2... in group2), ()
    files = glob.glob(dataset_path + 'data*') # cifar-10 data are named as data_batch_*
    for i in range(len(mixing)):
        filename = work_path + 'classgroup_{}'.format(i)
        target_class = mixing[i]
        labels = []
        images = []
        print(target_class)
        for f in files:
            current = cifar_unpickle(f)
            current_labels = current[b'labels']
            current_images = current[b'data']
            for index in range(len(current_labels)):
                if current_labels[index] in target_class:
                    labels.append(current_labels[index])
                    images.append(current_images[index])
        
        labels, images = prepare_format(labels, images)
        np.save(filename + 'label', labels)
        np.save(filename + 'image', images)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, work_path, nb_groups):
        self.labels = work_path + 'classgroup_{}'.format(nb_groups) + 'label'
        self.images = work_path + 'classgroup_{}'.format(nb_groups) + 'class'
        self.labels = np.load(self.labels)
        self.images = np.load(self.images)

    def __getitem__(self, index):
        return self.labels[index], self.images[index]
    
    def __len__(self):
        assert self.labels.shape[0] == self.images.shape[0]
        return len(self.labels.shape[0])
