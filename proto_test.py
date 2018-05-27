import pickle
import utils_data
import utils_resnet
import utils_icarl
import time
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader

def set_param():
    ######### Modifiable Settings ##########
    # load all params into dict for convience
    batch_size = 64                    # Batch size
    nb_val = 20                        # Validation sample per class
    nb_cl = 2                          # Classes per group
    nb_group = 5                       # Number of groups
    nb_proto = 20                      # Number of prototypes per class
    epochs = 3                        # Total number of epochs
    lr = 0.001                         # Initial learning rate
    lr_milestones = [4,8,12,16,20]   # Epochs where learning rate gets decreased
    lr_factor = 0.05                   # Learning rate decrease factor
    gpu = True                        # Use gpu for training
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
    return param

def set_path():
    ######### Paths  ##########
    # Working space
    # dataset_path = "d:/dataset/cifar-10-python"
    # testset_path = 'd:/ilte'
    # work_path = 'd:/ilex'
    # dataset_path = "/home/spyisflying/dataset/cifar/cifar-10-batches-py"
    # work_path = '/home/spyisflying/ilex'
    dataset_path = "/home/spyisflying/dataset/cifar/cifar-10-python"
    work_path = '/home/spyisflying/ilex'
    ###########################
    path = {'dataset_path': dataset_path, 'work_path': work_path}
    return path

def set_data(param, path):
    # Read label and random mixing
    label_name, label_dict = utils_data.parse_meta(path['dataset_path'])
    mixing = utils_data.generate_mixing(param['nb_group'], param['nb_cl'])
    # fix mixing for testing
    mixing = [(4, 7), (8, 5), (6, 2), (1, 3), (9, 0)] 
    print('Mixed class sequence: ')
    print(mixing)
    ### Preparing the files for the training/validation ###
    print("Creating training/validation data")
    # run once for specific mixing
    # utils_data.prepare_files_sample(dataset_path, work_path, mixing, nb_group, nb_cl, nb_val)
    return label_dict, mixing

def train_model(param, path, mixing):
    ### Start of the main algorithm ###
    print('apply training algorithm...')
    # Model initialization
    icarl = utils_icarl.iCaRL(param, label_dict)
    loss_fn = torch.nn.BCELoss(size_average = False)

    # Recording traing process in log file
    log = open(path['work_path'] + '/log.txt', 'ab', 0)
    log.write('epoch time training_loss validation_loss \n'.encode())

    # Training algorithm
    for iter_group in range(1): #nb_group
        # Training tools
        optimizer = torch.optim.Adam(icarl.parameters(), lr = param['lr'], weight_decay = param['wght_decay'])
        # scheduler = MultiStepLR(optimizer, milestones = lr_milestones, gamma = lr_factor)
        
        # Loading protoset
        if(iter_group == 0):
            protoset = dict()
            icarl_pre = None
        else:
            protoset_name = work_path + '/protoset_{}'.format(iter_group - 1)
            icarl_pre_name = work_path + '/model_{}'.format(iter_group - 1)
            with open(protoset_name, 'rb') as f:
                protoset = pickle.load(f)
                f.close()
            icarl_pre = torch.load(icarl_pre_name)
        # Loading trainging data by group
        data = utils_data.MyDataset(path['work_path'], iter_group, 0, protoset)
        loader = DataLoader(data, batch_size = param['batch_size'], shuffle = True)
        # Loading validation data by group
        data_val = utils_data.MyDataset(path['work_path'], iter_group, 1)
        loader_val = DataLoader(data_val, batch_size = param['batch_size'], shuffle = True)
        for epoch in range(param['epochs']):
            start = time.time()
            # Train
            error_train, error_val = 0, 0
            error_train = utils_icarl.train(icarl, icarl_pre, optimizer, loss_fn, loader)
            # Validate
            error_val = utils_icarl.val(icarl, icarl_pre, loss_fn, loader_val)
            # Print monitor info
            current_line = [epoch, time.time() - start, error_train / 600, error_val / 20]
            print(current_line)
            current_line = str(current_line)[1:-1] + '\n'
            log.write(current_line.encode())
            print('complete {}% on group {}'.format((epoch + 1) * 100 / param['epochs'], iter_group))
            # Save model every group_iter for babysitting model
            icarl_copy = copy.deepcopy(icarl)
            torch.save(icarl_copy, path['work_path'] + '/model_{}_{}'.format(iter_group, epoch))

def proto_test(icarl, protoset, iter_group, mixing, loader):
    print('Constructing protoset')
    protoset = icarl.construct_proto(iter_group, mixing, loader, protoset)
    print('Complete protoset')

    print('Testing')
    testset = utils_data.MyDataset(path['work_path'], 0, 2)
    testloader = DataLoader(testset, batch_size = param['batch_size'], shuffle = False)
    feature_mem, label_mem = icarl.feature_extract(testloader)
    result = icarl.classify(protoset, feature_mem, label_mem, iter_group)
    print('Complete test')
    return result

if __name__ == '__main__':
    param = set_param()
    path = set_path()
    label_dict, mixing = set_data(param, path)
#     train_model(param, path, mixing)
    for iter_group in range(1):
        protoset = dict()
        data = utils_data.MyDataset(path['work_path'], iter_group, 0, protoset)
        loader = DataLoader(data, batch_size = param['batch_size'], shuffle = True)
        result_mem = []
        for epoch in range(param['epochs']):
            icarl = torch.load(path['work_path'] + '/model_{}_{}'.format(iter_group, epoch))
            if(icarl.gpu):
                icarl = icarl.cuda()
            proto_test(icarl, dict(), iter_group, mixing, loader)
            current_result = list()
            for i in range(2):
                result = proto_test(icarl, protoset, iter_group, mixing, loader)
                current_result.append(result)
            result_mem.append(current_result)
        with open(path['work_path'] + '/result_{}'.format(iter_group), 'wb') as f:
            pickle.dump(result, f)
            f.close()