import pickle
import numpy as np
import glob
import torch.utils.data
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import random

def cifar_unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
        return dict

def parse_meta(dataset_path):
    label_name = cifar_unpickle(dataset_path + "/batches.meta")[b'label_names']
    label_name = [i.decode('utf-8') for i in label_name]
    label_dict = dict((i, label_name[i]) for i in range(len(label_name)))
    return label_name, label_dict

def generate_mixing(nb_groups, nb_cl):
    temp = list(range(nb_groups * nb_cl))
    random.shuffle(temp)
    idx = list(range(len(temp)))
    ans = []
    for i in idx[0 : len(idx) : nb_cl]:
        ans.append(tuple(temp[i : i + nb_cl]))
    return ans

def prepare_format(labels, images):
    labels = np.array(labels, dtype = np.uint8)
    images = np.array(images)
    images = images.reshape(images.shape[0], 32 * 32, 3)
    images = images.reshape(images.shape[0], 3 , 32, 32).transpose([0, 2, 3, 1])
    return labels, images

def prepare_files(dataset_path, work_path, mixing):
    # prepare image/label files according to mixing, who
    # will be load into model using dataloader
    # mixing format: [(class1, class2... in group1),
    # (class1, class2... in group2), ()
    files = glob.glob(dataset_path + '/data*') # cifar-10 data are named as data_batch_*
    for i in range(len(mixing)):
        filename = work_path + '/group_{}'.format(i)
        target_class = mixing[i]
        labels = []
        images = []
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
    def __init__(self, work_path, iter_group):
        super(MyDataset, self).__init__()
        self.labels = work_path + '/group_{}'.format(iter_group) + 'label.npy'
        self.images = work_path + '/group_{}'.format(iter_group) + 'image.npy'
        self.labels = np.load(self.labels)
        self.images = np.load(self.images)
        self.transform = data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
             ])
        
    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        image = Image.fromarray(image).resize((224, 224), Image.ANTIALIAS)
        image = self.transform(image)
        return image, label
    
    def __len__(self):
        assert self.labels.shape[0] == self.images.shape[0]
        return self.labels.shape[0]