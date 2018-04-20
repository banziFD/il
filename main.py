import utils_data
import utils_resnet
import utils_icarl
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
######### Modifiable Settings ##########
# load all params into dict for convience
batch_size = 16
nb_val = 50
nb_cl = 2
nb_group = 5
nb_proto = 20
epochs = 60
lr_old = 2
lr_start = [20, 30, 40, 50]
lr_fractor = 5
gpu = False
wght_decay = 0.00001
param = {
    'batch_size': batch_size,            # Batch size
    'nb_val': 50,                # Validation sample per class
    'nb_cl': 2,                 # Classes per group
    'nb_group': 5,             # Number of groups
    'nb_proto': 20,              # Number of prototypes per class
    'epoch': 60,                # Total number of epochs
    'lr_old': 2,                 # Initial learning rate
    'lr_start': [20, 30, 40, 50],# Epochs where learning rate gets decreased
    'lr_factor': 5,              # Learning rate decrease factor
    'gpu': False,                # Use gpu for training
    'wght_decay': 0.00001        # Weight Decay
}
########################################

######### Paths  ##########
# Working space
dataset_path = "/mnt/e/dataset/cifar-10-python"
work_path = '/mnt/e/ilex'

# dataset_path = "/home/spyisflying/dataset/cifar/cifar-10-batches-python"
# work_path = '/home/spyisflying/ilex'
###########################

# Read label and random mixing
label_name, label_dict = utils_data.parse_meta(dataset_path)
mixing = utils_data.generate_mixing(nb_group, nb_cl)
# fix mixing for testing
mixing = [(4, 7), (8, 5), (6, 2), (1, 3), (9, 0)] 
print('Mixed class sequence: ')
print(mixing)

### Preparing the files for the training/validation ###
print("Creating training/validation data")
# run once for specific mixing
#utils_data.prepare_files(dataset_path, work_path, mixing, nb_group, nb_cl)

### Initialization of some variables ###
class_means = np.zeros((512, nb_group * nb_cl, 2, nb_group))
loss_batch = []
files_protoset = [] * (nb_group * nb_cl)

### Start of the main algorithm ###
print('apply training algorithm...')
feature_net = utils_resnet.Resnet(pretrained = True)
icarl = utils_icarl.iCaRL(param, feature_net, label_dict)
loss_fn = torch.nn.BCELoss(size_average = False)
optimizer = torch.optim.Adam(icarl.parameters(), lr = 0.001, weight_decay = wght_decay)
if(gpu):
    icarl = icalr.cuda()
    loss_fn = loss_fn.cuda()

for iter_group in range(1): #nb_group
    # iter_group contrals 
    # loading data by group
    data = utils_data.MyDataset(work_path, iter_group)
    loader = DataLoader(data, batch_size = batch_size, shuffle = True)
    # known_mask
    known = Variable(icarl.known.clone(), requires_grad = False)
    # unknown_mask
    unknown = Variable(icarl.unknown.clone(), requires_grad = False)
    for epoch in range(param['epoch']):
        print('training on {} epoch'.format(epoch))
        for step, (x, y) in enumerate(loader):
            start = time.time()
            x = Variable(x)
            y = Variable(y.float(), requires_grad = False)
            if(gpu):
                x = x.cuda()
                y = y.cuda()
            y_pred = icarl(x)
            ### loss function ###
            # classification term + distillation term
            y_target = unknown * y + known * y_pred.detach()
            y_target = y_target.detach()
            loss = loss_fn(y_pred, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.data[0], ' time:', time.time() - start)

        print('complete {}% on group {}'.format(ephch / param['epoch'], iter_group))
    torch.save(icral, work_path+'/model')

    # Reduce Exemplar Set

    # Construct Examplar Set          