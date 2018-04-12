import torch
import torchvision.models as models

class ResNetFeature:
    def __init__(self, model = 'resnet18'):
        if(model == 'resnet18'):
            self.model = models.resnet18(pretrained = True)
            self.length = 512

        if(model == 'resnet34'):
            self.model = models.resnet34(pretrained = True)
            self.length = 512

        if(model == 'resnet50'):
            self.model = models.resnet50(pretrained = True)
            self.length = 2048

        if(model == 'resnet101'):
            self.model = models.resnet101(pretrained = True)
            self.length = 2048

        if(model == 'resnet152'):
            self.model = models.resnet152(pretrained = True)
            self.length = 2048

    def get_model(self):
        new_model = FeatureModule(slef.model, self.length)
        return new_model

class FeatureModule(torch.nn.Module):
    def __init__(self, base_model, length):
        self.base_model = list(base_model)[0:-1:1]
        self.base_model = torch.nn.Sequential(*(self.base_model))
        self.length = length

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        return x
