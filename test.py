import utils_data
import utils_icarl
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

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
test_path = '/mnt/e/ilte'

# dataset_path = "/home/spyisflying/dataset/cifar/cifar-10-batches-py"
# work_path = '/home/spyisflying/ilex'
# test_path = '/home/spyisflying/ilte'
###########################

mixing = [(4, 7), (8, 5), (6, 2), (1, 3), (9, 0)] 
model = torch.load(test_path + '/model19')
model.update_known(0, mixing)
print(model.class_mean[4])
print(model.class_mean[7])
# data = utils_data.MyDataset(work_path, 0, val = False, protoset = dict(), test = True)
# loader = DataLoader(data, batch_size = 1, shuffle = True)
# for step, (x, y, x_orig) in enumerate(loader):
#     x = Variable(x)
#     y_pred = model.classify(x)
