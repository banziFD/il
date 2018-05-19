import pickle
import numpy as np
import glob
import PIL.Image as Image
import random
import torch
import torch.utils.data
import torchvision.transforms as transforms

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

def prepare_format(labels, images, nb_group, nb_cl):
    labels = np.array(labels)
    images = np.array(images)
    images = images.reshape(images.shape[0], 32 * 32, 3)
    images = images.reshape(images.shape[0], 3 , 32, 32).transpose([0, 2, 3, 1])
    assert labels.shape[0] == images.shape[0]
    return labels, images

def prepare_files(dataset_path, work_path, mixing, nb_group, nb_cl, nb_val):
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
        labels, images = prepare_format(labels, images, nb_group, nb_cl)
        labels_val = labels[-nb_val:,]
        images_val = images[-nb_val:,]
        labels = labels[0:-nb_val,]
        images = images[0:-nb_val,]

        # save labels and images in workspace
        np.save(filename + 'label', labels)
        np.save(filename + 'image', images)
        np.save(filename + 'label_val', labels_val)
        np.save(filename + 'image_val', images_val)
    # prepare test files as above
    for i in range(len(mixing)):
        filename = work_path + '/group_{}'.format(i)
        target_class = mixing[i]
        current = cifar_unpickle(work_path + '/test_batch')
        current_labels = current[b'labels']
        current_images = current[b'data']
        labels_test = []
        images_test = []
        for index in range(len(current_labels)):
            if current_labels[index] in target_class:
                labels_test.append(current_labels[index])
                images_test.append(current_images[inedx])
        labels_test, iamges_test = prepare_format(labels, images, nb_group, nb_cl)
        np.save(filename + 'label_test', labels_test)
        np.save(filename + 'image_test', iamges_test)


def prepare_files_sample(dataset_path, work_path, mixing, 
nb_group, nb_cl, nb_val):
    ### Due to limitation on computing source, using sampled dataset to find hyper parameter ###
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
        labels, images = prepare_format(labels, images, nb_group, nb_cl)
        idx = np.random.choice(np.arange(labels.shape[0]), 
        600 + nb_val, replace = False)
        labels = labels[idx]
        images = images[idx]
        labels_val = labels[-nb_val:,]
        images_val = images[-nb_val:,]
        labels = labels[0:-nb_val,]
        images = images[0:-nb_val,]
        np.save(filename + 'label', labels)
        np.save(filename + 'image', images)
        np.save(filename + 'label_val', labels_val)
        np.save(filename + 'image_val', images_val)
    for i in range(len(mixing)):
        filename = work_path + '/group_{}'.format(i)
        target_class = mixing[i]
        current = cifar_unpickle(dataset_path + '/test_batch')
        current_labels = current[b'labels']
        current_images = current[b'data']
        labels_test = []
        images_test = []
        for index in range(len(current_labels)):
            if current_labels[index] in target_class:
                labels_test.append(current_labels[index])
                images_test.append(current_images[index])
        labels_test, iamges_test = prepare_format(labels_test, 
        images_test, nb_group, nb_cl)
        np.save(filename + 'label_test', labels_test)
        np.save(filename + 'image_test', iamges_test)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, work_path, iter_group, mode = 0, protoset = dict()):
        ### mode (0 stands for train + protoset)
        ###      (1 stands for val) 
        ###      (2 stands for test) 
        ###      (3 stands for protoset only)
        super(MyDataset, self).__init__()
        self.mode = mode
        if(mode == 0):
            self.labels = work_path + '/group_{}'.format(iter_group) + 'label.npy'
            self.images = work_path + '/group_{}'.format(iter_group) + 'image.npy'
            self.labels = np.load(self.labels)
            self.images = np.load(self.images)
            if(any(protoset)):
                for cl in protoset:
                    proto_labels = np.ones(20, dtype = int) * cl
                    proto_images = protoset[cl][1]                   
                    self.images = np.concatenate((self.images, proto_images), 
                    axis = 0)
                    self.labels = np.concatenate((self.labels, proto_labels), 
                    axis = 0)
        if(mode == 1):
            self.labels = work_path + '/group_{}'.format(iter_group) + 'label_val.npy'
            self.images = work_path + '/group_{}'.format(iter_group) + 'image_val.npy'
            self.labels = np.load(self.labels)
            self.images = np.load(self.images)
        if(mode == 2):
            self.labels = work_path + '/group_{}'.format(iter_group) + 'label_test.npy'
            self.images = work_path + '/group_{}'.format(iter_group) + 'image_test.npy'
            self.labels = np.load(self.labels)
            self.images = np.load(self.images)
        if(mode == 3):
            for cl in protoset:
                proto_labels = np.ones(20) * cl
                proto_images = protoset[cl][1]
                self.images = proto_images
                self.labels = proto_labels      
                
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
             ])

    def __getitem__(self, index):
        label = self.labels[index]
        # label: [[0 1 0 0 ...]...]
        label_sca = label
        label = torch.zeros(10)
        label[label_sca] = 1
        image = self.images[index]
        image_orig = image
        image = Image.fromarray(image).resize((224, 224), Image.ANTIALIAS)
        image = self.transform(image)
        return image, label, image_orig, label_sca
    
    def __len__(self):
        assert self.labels.shape[0] == self.images.shape[0]
        return self.labels.shape[0]
