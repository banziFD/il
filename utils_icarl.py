import torch

class iCaRL(torch.nn.Module):
    def __init__(self, param, feature_net, label_dict):
        super(iCaRL, self).__init__()
        # self.files_protoset = params['files_protoset']
        self.nb_class = param['nb_cl']
        self.nb_group = param['nb_group']
        self.label_dict = label_dict
        total_cl = param['nb_cl'] * param['nb_group']
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
        model = nn.Sequential(list(self.children())[0:-2:1])
