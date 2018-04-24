import numpy as np
import torch
from torch.autograd import Variable

class iCaRL(torch.nn.Module):
    def __init__(self, param, feature_net, label_dict):
        super(iCaRL, self).__init__()
        self.nb_class = param['nb_cl']
        self.nb_group = param['nb_group']
        self.label_dict = label_dict
        self.nb_proto = param['nb_proto']
        total_cl = param['nb_cl'] * param['nb_group']
        self.class_mean = torch.zeros(total_cl, 512)
        self.known = torch.zeros(total_cl)
        self.unknown = torch.ones(total_cl)
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
 
    def classify(self, x):
        # nearest-mean-of-examplars classification based on
        # feature map extracted by resnet
        feature = self.feature_net(x)
        feature = feature.data
        assert self.class_mean.shape[0] == self.known.shape[0]
        d = torch.Tensor(float('Inf'))
        ans = -1
        for i in range(self.class_mean.shape[0]):
            if(self.known[i] == 1):
                new_d = torch.dist(feature, class_mean[i])
                if(d > torch.dist()):
                    ans = i
        assert i != -1
        return ans
    
    def update_mean(self, protoset):
        pass
    
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
        protoset = image_mem[index]
        return protoset
    
    def construct_proto(self, iter_group, mixing, loader, protoset):
        feature_net = self.feature_net
        current_cl = mixing[iter_group]
        feature_mem = dict()
        image_mem = dict()
        protoset = protoset
        # Initialize feature_mem / image_mem dict
        for i in current_cl:
            feature_mem[i] = torch.zeros(1, 512)
            image_mem[i] = torch.zeros(1, 32, 32, 3)
        for step, (x, y, x_orig) in enumerate(loader):
            x = Variable(x, requires_grad = False)
            y = y.nonzero()
            # feature = feature_net(x)
            # feature = feature.data
            # feature = feature.view(feature.size(0), -1)
            feature = torch.rand(16, 512)
            x_orig = x_orig.float()
            for item in y:
                if(item[1] in current_cl):
                    feature_mem[item[1]] = torch.cat((feature_mem[item[1]], feature[item[0]].view(1, -1)), 0)
                    image_mem[item[1]] = torch.cat((image_mem[item[1]], x_orig[item[0]].view(1, 32, 32, 3)), 0)
        for i in current_cl:
            feature_mem[i] = feature_mem[i][1:, :]
            image_mem[i] = image_mem[i][1:, :, :, :]
        # Update classmean for classify
        for i in current_cl:
            mean = torch.mean(feature_mem[i], 0)
            self.class_mean[i] = mean
        # Choose image as protoset example
        nb_proto = self.nb_proto
        for i in current_cl:
            protoset[i] = self.choose_top(nb_proto, feature_mem[i], image_mem[i], self.class_mean[i])
            protoset[i] = protoset[i].numpy().astype(np.uint8)
        return protoset
