import utils_data
import utils_icarl
import pickle
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
#utils_data.prepare_files_sample(dataset_path, work_path, mixing, nb_group, nb_cl, nb_val)
protoset = pickle.load(open(test_path + '/protoset_0_19','rb'))
testset = utils_data.MyDataset(test_path, 0, 2, protoset)
model = torch.load(test_path + '/model0_19')
loader = DataLoader(testset, batch_size = 16, shuffle = False)
#model.feature_extract(loader, test_path)
model.classify(protoset, test_path)