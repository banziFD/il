import torch

class iCaRL(torch.nn.Module):
    def __init__(self, param, feature_net, label_dict):
        super(iCaRL, self).__init__()
        # self.files_protoset = params['files_protoset']
        self.nb_class = param['nb_cl']
        self.nb_group = param['nb_group']
        self.label_dict = label_dict
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
        assert self.class_mean.shape[0] = self.known.shape[0]
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
        for file_name in protoset:
            pass

    def update_known(self, iter_group, mixing):
        for known_cl in mixing[iter_group]:
            self.known[known_cl] = 1
            self.unknown[known_cl] = 0

    def construct_proto(self, iter_group, mixing, loader, proteset):
        feature_net = self.feature_net
        current_cl = mixing[iter_group]
        feature_mem = dict()
        image_mem = dict()
        protoset = dict()
        # Initialize feature_mem / image_mem dict
        for i in current_cl:
            feature_mem[i] = torch.Tensor([])
            image_mem[i] = torch.Tensor([])
        for step, (x, y, x_orig) in enumerate(loader):
            x = Variable(x, requires_grad = False)
            y = y.nonzero()
            feature = feature_net(x)
            feature = feature.data
            feature = feature.view(feature.size(0), -1)
            for item in y:
                feature_mem[item[1]] = torch.cat((feature_mem[item[1]], feature[item[0]]), 0)
                image_mem[item[i]] = torch.cat((image_mem[item[1]], x_orig[item[0]),0)
        
        # Update classmean for classify
        for i in current_cl:
            mean = torch.mean(feature_mem[i], 0)
            self.class_mean[i] = mean

        # Choose image as protoset example
        nb_proto = param['nb_proto']
        for i in current_cl:
            protoset[i] = choose_top(nb_proto, feature_mem[i], image[i])
        
        return protoset

    def choose_top(self, nb_proto, feature_mem, image, class_mean):
        assert feature.shape[0] == iamge.shape[0]
        distance = feature_mem - class_mean
        distance = distance * distance
        distance = torch.sum(distance, 1)
        value, index = torch.topk(distance, nb_proto, largest = False)
        protoset = image[index]
        return protoset
    
    def save_proto(self):
        pass