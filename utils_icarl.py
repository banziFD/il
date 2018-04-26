import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

def train(icarl, optimizer, scheduler, loss_fn, loader):
    # known_mask
    known = Variable(icarl.known.clone(), requires_grad = False)
    # unknown_mask
    unknown = Variable(icarl.unknown.clone(), requires_grad = False)
    scheduler.step()
    error_train = 0
    for step, (x, y) in enumerate(loader):
        x = Variable(x)
        y = Variable(y.float(), requires_grad = False)
        if(icarl.gpu):
            x = x.cuda()
            y = y.cuda()
        y_pred = icarl(x)
        ### loss function ###
        # classification term + distillation term
        y_target = unknown * y + known * y_pred.detach()
        y_target = y_target.detach()
        loss = loss_fn(y_pred, y_target)
        # backword and update model
        optimizer.zero_grad()
        error_train = error_train + loss.data[0]
        loss.backward()
        optimizer.step()
    return error_train
        
def val(icarl, loss_fn, loader_val):
    # known_mask
    known = Variable(icarl.known.clone(), requires_grad = False)
    # unknown_mask
    unknown = Variable(icarl.unknown.clone(), requires_grad = False)
    error_val = 0
    for step, (x, y) in enumerate(loader_val):
        x = Variable(x, requires_grad = False)
        y = Variable(y.float(), requires_grad = False)
        if(icarl.gpu):
            x = x.cuda()
            y = y.cuda()
        y_pred = icarl(x)
        y_target = unknown * y + known * y_pred.detach()
        y_target = y_target.detach()
        loss_val = loss_fn(y_pred, y_target)
        error_val = error_val + loss_val.data[0]      
    return error_val

class iCaRL(torch.nn.Module):
    def __init__(self, param, feature_net, label_dict):
        super(iCaRL, self).__init__()
        self.total_cl = param['nb_cl'] * param['nb_group']
        self.label_dict = label_dict
        self.nb_proto = param['nb_proto']
        self.gpu = param['gpu']
        self.total_cl = param['nb_cl'] * param['nb_group']
        self.known = torch.zeros(self.total_cl)
        self.unknown = torch.ones(self.total_cl)
        self.feature_net = feature_net
        self.linear = torch.nn.Linear(512, total_cl)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # extract freature map by resnet
        y = self.feature_net(x)
        # compute sigmoid value for every class
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        y = self.sigmoid(y)
        return y

    def classify(self, protoset, test_path):
        known_cl = protoset.keys()
        class_mean = tensor.zeros(self.total_cl, 512)
        for cl in known_cl:
            proto_image = protoset[cl]
            x = Variable(proto_image, requires_grad = False)
            feature = feature_net(x)
            feature = feature.data
            feature = feature.view(self.nb_proto, -1)
            mean = torch.mean(feature, 0)
            class_mean[cl] = mean
        feature = torch.load(test_path + '/feature')
        label = torch.load(test_path + '/label')
        assert feature.shape[0] == label.shape[0]
        count_true = 0
        count_all = feature.shape[0]
        for index in range(count_all):
            current_f = feature[index]
            current_l = label[index]
            distance = class_mean - current_f
            distance = distance * distance
            distance = torch.sum(distance, 1)
            dist, y_pred = torch.topk(distance, 1, largest = False)
            if(y_pred == label[index]):
                count_true += 1
        print(count_true / count_all)
            

    def feature_extract(self, loader, test_path):
        # nearest-mean-of-examplars classification based on
        # feature map extracted by resnet
        feature_net = self.feature_net
        test_count = 0
        for step, (x, y, y_sca) in enumerate(loader):
            test_count += 1

        label_mem = torch.zeros(test_count, self.total_cl)
        feature_mem = torch.zeros(test_count, 512)
        test_count = 0
        for step, (x, y, y_sca) in enumerate(loader):
            x = Variable(x, requires_grad = False)
            feature = feature_net(x)
            feature = feature.data
            feature = feature.view(feature.shape[0], -1)
            for i in range(feature.shape[0]):
                label_mem[test_count] = y_sca[i]
                feature_mem[test_count] = feature[i]
                test_count += 1
        torch.save(feature_mem, test_path + '/feature')
        torch.save(label_mem, test_path + '/label')

    def update_known(self, iter_group, mixing):
        for known_cl in mixing[iter_group]:
            self.known[known_cl] = 1
            self.unknown[known_cl] = 0
    
    # Pick up top-nb_proto images as example of typical class
    def choose_top(self, nb_proto, feature_mem, image_mem, class_mean):
        assert feature_mem.shape[0] == image_mem.shape[0]
        distance = feature_mem - class_mean
        distance = distance * distance
        distance = torch.sum(distance, 1)
        value, index = torch.topk(distance, nb_proto, largest = False)
        protoset = image_mem[index].clone()
        return protoset
    
    def construct_proto(self, iter_group, mixing, loader, protoset):
        feature_net = self.feature_net
        new_cl = mixing[iter_group]
        feature_mem = dict()
        image_mem = dict()
        class_mean = dict()
        class_count = dict()
        protoset = protoset
        # Extract features and save it into feature_mem
        for step, (x, y, y_sca) in enumerate(loader):
            for cl in y_sca:
                if cl in class_count:
                    class_count[cl] += 1
                else:
                    class_count[cl] = 1
        
        for cl in class_count:
            feature_mem[cl] = torch.zero(class_count[cl], 512)
            image_mem[cl] = torch.zeros(class_count[cl], 3, 224, 224)
        
        for step, (x, y, y_sca) in enumerate(loader):
            x = Variable(x, requires_grad = False)
            # feature = feature_net(x)
            # feature = feature.data
            # feature = feature.view(feature.shape[0], -1)
            feature = torch.rand(x.shape[0], 512)
            x = x.data
            for index in range(y.shape[0]):
                cl = y[index]
                feature_mem[cl][class_count[cl] - 1] = feature[index]
                image_mem[cl][class_count[cl] - 1] = x[index]
                class_count[cl] -= 1
                
        # Update classmean for classify
        for cl in new_cl:
            mean = torch.mean(feature_mem[cl], 0)
            class_mean[cl] = mean
        
        # Choose image as protoset example
        nb_proto = self.nb_proto
        for cl in new_cl:
            protoset[cl] = self.choose_top(nb_proto, feature_mem[cl], image_mem[cl], class_mean[cl])
            protoset[cl] = protoset[cl].numpy().astype(np.uint8)
        return protoset
