import torch
import torchvision

# Use resnet18 model from torchvision as feature extractor.(details at
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
# For iCaRL, you can actually try different CNN structures.

class Resnet(torch.nn.Module):
    def __init__(self, pretrained = False):
        super(Resnet, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained = pretrained)
        layers = list(resnet18.children())
        # delete last fully connected layer
        layers = layers[0:-1:1]
        self.resnet = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # input Variable shape: (n, channels, 224, 224)
        # output feature shape: (n, 512, 1, 1) / (n, 512)
        y_feature = self.resnet(x)
        return y_feature
