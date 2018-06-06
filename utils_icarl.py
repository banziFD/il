import numpy as np
import torch
import utils_resnet

def train(icarl, icarl_pre, optimizer, loss_fn, loader):
    # known_mask & unknown_mask
    known = icarl.known.clone()
    unknown = icarl.unknown.clone()
    error_train = 0
    for step, (x, y, x_orig, y_sca) in enumerate(loader):
        x.requires_grad = False
        y.requires_grad = False
        # seperate known classes with unknown classes
        known_count = int(torch.sum(y * known).item())
        unknown_count = x.shape[0] - known_count
        x_known = torch.zeros(known_count, 3, 224, 224)
        y_known = torch.zeros(known_count, 10)
        x_unknown = torch.zeros(unknown_count, 3, 224, 224)
        y_unknown = torch.zeros(unknown_count, 10)
        i, j = 0, 0
        for k in range(y.shape[0]):
            if(known[y_sca[k]] == 1):
                x_known[i] = x[k]
                y_known[i] = y[k]
                i += 1
            else:
                x_unknown[j] = x[k]
                y_unknown[j] = y[k]
                j += 1

        # Load data to gpu memory if cuda is availiable
        if(icarl.gpu):
            icarl = icarl.cuda()
            if (icarl_pre != None):
                icarl_pre = icarl_pre.cuda()
            loss_fn = loss_fn.cuda()
            x_known = x_known.cuda()
            y_known = y_known.cuda()
            x_unknown = x_unknown.cuda()
            y_unknown = y_unknown.cuda()
        # Forward prop, modified x means we reorder original images
        # by (known, unknown)
        x_modify = torch.cat((x_known, x_unknown), 0)
        y_pred = icarl(x_modify)

        ### loss function ###
        # classification term + distillation term
        if(icarl_pre != None and known_count != 0):
            y_diss = icarl_pre(x_known)
            y_diss = y_diss.detach()
            y_class = y_unknown
            y_target = torch.cat((y_diss, y_class), 0)
        else:
            y_target = y_unknown
        loss = loss_fn(y_pred, y_target)

        # backprop and update model
        optimizer.zero_grad()
        error_train = error_train + loss.data.item()
        loss.backward()
        optimizer.step()
    if(icarl.gpu):     
        # Save any model in cpu mode so it work cross all platform
        icarl = icarl.cpu()
    return error_train
        
def val(icarl, icarl_pre, loss_fn, loader_val):
    # known_mask & unknown_mask
    known = icarl.known.clone()
    unknown = icarl.unknown.clone()
    error_val = 0
    for step, (x, y, x_orig, y_sca) in enumerate(loader_val):
        x.requires_grad = False
        y.requires_grad = False
        # sperate known classes with unknown classes
        known_count = int(torch.sum(y * known,).item())
        unknown_count = x.shape[0] - known_count
        x_known = torch.zeros(known_count, 3, 224, 224)
        y_known = torch.zeros(known_count, 10)
        x_unknown = torch.zeros(unknown_count, 3, 224, 224)
        y_unknown = torch.zeros(unknown_count, 10)
        i, j = 0, 0
        for k in range(y.shape[0]):
            if(known[y_sca[k]] == 1):
                x_known[i] = x[k]
                y_known[i] = y[k]
                i += 1
            else:
                x_unknown[j] = x[k]
                y_unknown[j] = y[k]
                j += 1
        
        # Load data to gpu memory if cuda is availiable
        if(icarl.gpu):
            icarl = icarl.cuda()
            if(icarl_pre != None):
                icarl_pre = icarl_pre.cuda()
            loss_fn = loss_fn.cuda()
            x_known = x_known.cuda()
            y_known = y_known.cuda()
            x_unknown = x_unknown.cuda()
            y_unknown = y_unknown.cuda()
        # Forward prop, modified x means we reorder original images
        # by (known, unknown)
        x_modify = torch.cat((x_known, x_unknown), 0)
        y_pred = icarl(x_modify)

        ### loss function ###
        # classification term + distillation term
        if(icarl_pre != None and known_count != 0):
            y_diss = icarl_pre(x_known)
            y_diss = y_diss.detach()
            y_class = y_unknown
            y_target = torch.cat((y_diss, y_class), 0)
        else:
            y_target = y_unknown
        loss = loss_fn(y_pred, y_target)
        error_val = error_val + loss.data.item()
    if(icarl.gpu):
        icarl = icarl.cpu()
    return error_val    
        
        

class iCaRL(torch.nn.Module):
    def __init__(self, param, label_dict):
        super(iCaRL, self).__init__()
        self.total_cl = param['nb_cl'] * param['nb_group']
        self.label_dict = label_dict
        self.nb_proto = param['nb_proto']
        self.gpu = param['gpu']
        self.total_cl = param['nb_cl'] * param['nb_group']

        self.known = torch.zeros(self.total_cl, requires_grad = False)
        self.unknown = torch.ones(self.total_cl, requires_grad = False)
        self.feature_net = utils_resnet.Resnet(pretrained = True)
        self.linear = torch.nn.Linear(512, self.total_cl)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # extract freature map by resnet
        y = self.feature_net(x)
        # compute sigmoid value for every class
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        y = self.sigmoid(y)
        return y

    def classify(self, protoset, feature_mem, label_mem, iter_group):
        known_cl = protoset.keys()
        class_mean = torch.zeros(self.total_cl, 512, requires_grad = False)
        # compute current mean feature for each class
        for cl in known_cl:
            proto_image = protoset[cl][0]
            if(self.gpu):
                proto_image = proto_image.cuda()
            x = proto_image
            x.requires_grad = False
            mean = self.feature_net(x).detach()
            mean = torch.mean(mean, 0)
            class_mean[cl] = mean
#             class_mean[cl] = protoset[cl][2]
        
        assert feature_mem.shape[0] == label_mem.shape[0]
        count_true = 0
        count_all = feature_mem.shape[0]
        for index in range(count_all):
            current_f = feature_mem[index]
            current_l = label_mem[index]
            distance = class_mean - current_f
            distance = distance * distance
            distance = torch.sum(distance, 1)
            dist, y_pred = torch.topk(distance, 1, largest = False)
            y_pred = y_pred.int()
            if(y_pred[0] == label_mem[index]):
                count_true += 1
        print('Accuracy: ',count_true / count_all)
        return count_true / count_all
            
    def feature_extract(self, loader):
        # nearest-mean-of-examplars classification based on
        # feature map extracted by resnet
        feature_net = self.feature_net
        test_count = 0
        for step, (x, y, x_orig, y_sca) in enumerate(loader):
            test_count += y_sca.shape[0]
        # Pre-allocate memory 
        label_mem = torch.zeros(test_count, requires_grad = False)
        feature_mem = torch.zeros(test_count, 512, requires_grad = False)
        if(self.gpu):
            label_mem = label_mem.cuda()
            feature_mem = feature_mem.cuda()
        
        # Extract features and save into label_mem/feature_mem
        test_count = 0
        for step, (x, y, x_orig, y_sca) in enumerate(loader):
            x.requires_grad = False
            if(self.gpu):
                x = x.cuda()
                y_sca = y_sca.cuda()
            
            # Extract features
            feature = feature_net(x)
            feature = feature.detach()
            # Save features
            for i in range(feature.shape[0]):
                label_mem[test_count] = y_sca[i]
                feature_mem[test_count] = feature[i]
                test_count += 1
        label_mem = label_mem.int()
        # Move every data back to cpu before it being saved
        if(self.gpu):
            feature_mem = feature_mem.cpu()
            label_mem = label_mem.cpu()
        return feature_mem, label_mem

    def update_known(self, iter_group, mixing):
        for known_cl in mixing[iter_group]:
            self.known[known_cl] = 1
            self.unknown[known_cl] = 0
    
    # Pick up top-nb_proto images as example of typical class
    def choose_top_(self, nb_proto, feature_mem, 
    image_mem, image_orig_mem, class_mean):
        assert feature_mem.shape[0] == image_mem.shape[0]
        distance = feature_mem - class_mean
        distance = distance * distance
        distance = torch.sum(distance, 1)
        value, index = torch.topk(distance, nb_proto, largest = False)
        protoset = image_mem[index].clone()
        protoset_orig = image_orig_mem[index].clone()
        return [protoset, protoset_orig, class_mean]
    
    def choose_top(self, nb_proto, feature_mem, image_mem, image_orig_mem, class_mean):
        assert feature_mem.shape[0] == image_mem.shape[0]
        visited = torch.zeros(feature_mem.shape[0], dtype = torch.uint8)
        tot = torch.zeros(feature_mem.shape[1])
        if(self.gpu):
            tot = tot.cuda()
        protoset_index = []
        for i in range(nb_proto):
            distance = torch.tensor(float('inf'))
            if(self.gpu):
                distance = distance.cuda()
            p = -1
            for item in range(feature_mem.shape[0]):
                if(visited[item] == 0):
                    avg = (tot + feature_mem[i]) / (i + 1)
                    distance_temp = torch.norm(class_mean - avg)
                    if(distance_temp < distance):
                        p = item
                        distance = distance_temp
            protoset_index.append(p)
            visited[p] = 1
            tot = tot + feature_mem[p]
        index = torch.tensor(protoset_index)
        protoset = image_mem[index].clone()
        protoset_orig = image_orig_mem[index].clone()
        return [protoset, protoset_orig, class_mean]
               
    def construct_proto(self, iter_group, mixing, loader, protoset):
        # Protoset will be a dictionary of tuples, where keys are class labels, values are tuple of (image, image_orig, feature_mean)
        feature_net = self.feature_net
        new_cl = mixing[iter_group]
        feature_mem = dict()
        image_mem = dict()
        image_orig_mem = dict()
        class_mean = dict()
        class_count = dict()
        protoset = protoset

        # Extract features and save it into feature_mem
        for step, (x, y, x_orig, y_sca) in enumerate(loader):
            for cl in y_sca:
                cl = int(cl.item())
                if cl in class_count:
                    class_count[cl] += 1
                else:
                    class_count[cl] = 1
        # Pre-allocate memory
        for cl in class_count:
            feature_mem[cl] = torch.zeros(class_count[cl], 512).double()
            image_mem[cl] = torch.zeros(class_count[cl], 3, 224, 224)
            image_orig_mem[cl] = torch.zeros(class_count[cl], 32, 32, 3)
            if(self.gpu):
                feature_mem[cl] = feature_mem[cl].cuda()
                image_mem[cl] = image_mem[cl].cuda()
                image_orig_mem[cl] = image_orig_mem[cl].cuda()
                
        for step, (x, y, x_orig, y_sca) in enumerate(loader):
            if(self.gpu):
                x = x.cuda()
                x_orig = x_orig.cuda()
                y_sca = y_sca.cuda()
            feature = feature_net(x)
            feature = feature.detach()
            feature = feature.view(feature.shape[0], -1)
            for index in range(y.shape[0]):
                cl = y_sca[index].item()
                feature_mem[cl][class_count[cl] - 1] = feature[index]
                image_mem[cl][class_count[cl] - 1] = x[index]
                image_orig_mem[cl][class_count[cl] - 1] = x_orig[index]
                class_count[cl] -= 1
        # Compute class_mean for pick examplar
        for cl in new_cl:
            mean = torch.mean(feature_mem[cl], 0)
            class_mean[cl] = mean
        # Choose image as protoset example
        nb_proto = self.nb_proto
        for cl in new_cl:
            protoset[cl] = self.choose_top(nb_proto, feature_mem[cl], image_mem[cl],image_orig_mem[cl], class_mean[cl])
            if(self.gpu):
                protoset[cl][0] = protoset[cl][0].cpu()
                protoset[cl][1] = protoset[cl][1].cpu()
                protoset[cl][2] = protoset[cl][2].cpu()
            protoset[cl][1] = protoset[cl][1].numpy().astype(np.uint8)
            protoset[cl] = tuple(protoset[cl])
        return protoset
    