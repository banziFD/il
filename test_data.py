import utils_data
work_path = '/mnt/e/ilex/'
dataset_path = '/mnt/e/dataset/cifar-10-python/'
mixing = [(1, 2)]
#utils_data.prepare_files(dataset_path, work_path, mixing)
import numpy as numpy
import torch
s = utils_data.MyDataset(work_path, 0)
loader = torch.utils.data.DataLoader(s, batch_size = 64)
