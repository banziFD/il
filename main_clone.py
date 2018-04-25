import pickle
import utils_data
import utils_resnet
import utils_icarl
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
######### Modifiable Settings ##########
# load all params into dict for convience
batch_size = 16                    # Batch size
nb_val = 20                        # Validation sample per class
nb_cl = 2                          # Classes per group
nb_group = 5                       # Number of groups
nb_proto = 20                      # Number of prototypes per class
epochs = 20                        # Total number of epochs
lr = 1                             # Initial learning rate
lr_milestones = [3, 6, 9, 12, 15]  # Epochs where learning rate gets decreased
lr_factor = 0.05                    # Learning rate decrease factor
gpu = False                        # Use gpu for training
wght_decay = 0.00001               # Weight Decay
param = {
    'batch_size': batch_size,           
    'nb_val': nb_val,               
    'nb_cl': nb_cl,                 
    'nb_group': nb_group,             
    'nb_proto': nb_proto,              
    'epochs': epochs,                
    'lr': lr,                 
    'lr_milestones': lr_milestones,
    'lr_factor': lr_factor,              
    'gpu': gpu,                
    'wght_decay': wght_decay        
}
########################################

######### Paths  ##########
# Working space
dataset_path = "/mnt/e/dataset/cifar-10-python"
work_path = '/mnt/e/ilex'
# dataset_path = "/home/spyisflying/dataset/cifar/cifar-10-batches-py"
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
utils_data.prepare_files_sample(dataset_path, work_path, mixing, nb_group, nb_cl, nb_val)

### Initialization of some variables ###
class_means = np.zeros((512, nb_group * nb_cl, 2, nb_group))
loss_batch = []
files_protoset = [] * (nb_group * nb_cl)

### Start of the main algorithm ###
print('apply training algorithm...')

# Model initialization
feature_net = utils_resnet.Resnet(pretrained = True)
icarl = utils_icarl.iCaRL(param, feature_net, label_dict)

# Training tools
loss_fn = torch.nn.BCELoss(size_average = False)
optimizer = torch.optim.Adam(icarl.parameters(), lr = lr, weight_decay = wght_decay)
scheduler = MultiStepLR(optimizer, milestones = lr_milestones, gamma = 0.1)
if(gpu):
    icarl = icalr.cuda()
    loss_fn = loss_fn.cuda()
    scheduler = scheduler.cuda()

# Recording traing process
log = open(work_path + '/log.txt', 'ab', 0)
log.write('epoch time training_loss validation_loss \n'.encode())

for iter_group in range(2): #nb_group
    # Loading protoset
    if(iter_group == 0):
        protoset = dict()
    else:
        protoset_name = work_path + '/protoset{}'.format(iter_group - 1)
        with open(protoset_name, 'rb') as f:
            protoset = pickle.load(f)
            f.close()
            
    # Loading trainging data by group
    data = utils_data.MyDataset(work_path, iter_group, val = False, protoset = protoset)
    loader = DataLoader(data, batch_size = batch_size, shuffle = True)
    # loading validation data by group
    data_val = utils_data.MyDataset(work_path, iter_group, True)
    loader_val = DataLoader(data_val, batch_size = batch_size, shuffle = True)
    # known_mask
    known = Variable(icarl.known.clone(), requires_grad = False)
    # unknown_mask
    unknown = Variable(icarl.unknown.clone(), requires_grad = False)
    for epoch in range(param['epochs']):
        scheduler.step()
        start = time.time()
        error_train, error_val = 0, 0
        for step, (x, y, x_orig) in enumerate(loader):
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
            # backword and update model
            optimizer.zero_grad()
            error_train = error_train + loss.data[0]
            loss.backward()
            optimizer.step()
        for step, (x_, y_, x_orig) in enumerate(loader_val):
            x_ = Variable(x_, requires_grad = False)
            y_ = Variable(y_.float(), requires_grad = False)
            if(gpu):
                x_ = x_.cuda()
                y_ = y_.cuda()
            y_pred_ = icarl(x)
            y_target_ = unknown * y + known * y_pred.detach()
            y_target_ = y_target.detach()
            loss_val = loss_fn(y_pred, y_target)
            error_val = error_val + loss_val.data[0]      
        current_line = [epoch, time.time() - start, error_train / 600, error_val / 20]
        print(current_line)
        current_line = str(current_line)[1:-1] + '\n'
        log.write(current_line.encode())
        print('complete {}% on group {}'.format((epoch + 1) * 100 / epochs, iter_group))
        torch.save(icarl, work_path+'/model{}'.format(epoch))

    # Construct Examplar Set and save it as a dict
    loader = DataLoader(data, batch_size = batch_size, shuffle = True)
    size = len(data)
    protoset = icarl.construct_proto(iter_group, mixing, loader, protoset)
    icarl.update_known(iter_group, mixing)
    protoset_name = work_path + '/protoset{}'.format(iter_group)
    with open(protoset_name, 'wb') as f:
        pickle.dump(protoset, f)
        f.close()
log.close()