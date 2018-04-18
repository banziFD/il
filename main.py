import utils_data
import utils_icarl

######### Modifiable Settings ##########
# load all params into dict for convience
params = {
    'batch_size': 64,            # Batch size
    'nb_val': 50,                # Validation sample per class
    'nb_cl': 10,                 # Classes per group
    'nb_groups': 10,             # Number of groups
    'nb_proto': 20,              # Number of prototypes per class
    'epochs': 60,                # Total number of epochs
    'lr_old': 2,                 # Initial learning rate
    'lr_start': [20, 30, 40, 50] # Epochs where learning rate gets decreased
    'lr_factor': 5               # Learning rate decrease factor
    'gpu': False                 # Use gpu for training
    'wght_decay': 0.00001        # Weight Decay
}
########################################

######### Paths  ##########
# Working space
dataset_path = "E:\dataset\cifar-10-python"
work_path = "E:\ilex\"
###########################

### Initialization of some variables ###
class_means = np.zeros((512, nb_groups * nb_cl, 2, nb_groups))
loss_batch = []
files_protoset = []
for _ in range(nb_groups * nb_cl):
    files_protoset.append([])

# Random mixing
print("Mixing the classes and putting them in batches of classes...")
np.random.seed(1993)
order = np.arange(nb_groups * nb_cl)
mixing = np.arrange(nb_groups * nb_cl)
np.random.shuffle(mixing)

### Preparing the files for the training/validation ###
print("Creating training/ validation data）
utils.prepare_files(dataset_path, work_path, )

