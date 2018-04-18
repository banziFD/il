import torch.nn as nn

class iCaRL(nn.model):
    def __init__(self, params, resnet, pretrained = False):
        self.class_means = params['class_means'
        self.loss_batch = params['loss_batch']
        self.files_protoset = params['files_protoset']
        self.nb_class = params['nb_class']
        self.nb_group = params['nb_group']
        self.label_dict = param['label_dict']
        self.linear = nn.Linear(512, self.nb_class * self.nb_group)
        self.sigmoid = nn.Sigmoid()
        self.resnet = resnet
    
    def forward(self, x):
        # extract freature map by resnet
        y = self.resnet(x)
        # compute sigmoid value for every class
        y = self.linear(y)
        y = self.sigmoid(y)
        return y
 
    def classify(self, x):
        # nearest-mean-of-examplars classification based on
        # feature map extracted by resnet
        model = nn.Sequential(list(self.children())[:-2:]
        pass
    


    