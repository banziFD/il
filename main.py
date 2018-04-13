import pickle

######### Modifiable Settings ##########
batch_size = 128            # Batch size
nb_val     = 50             # Validation samples per class
nb_cl      = 10             # Classes per group 
nb_groups  = 10             # Number of groups
nb_proto   = 20             # Number of prototypes per class: total protoset memory/ total number of classes
epochs     = 60             # Total number of epochs 
lr_old     = 2.             # Initial learning rate
lr_strat   = [20,30,40,50]  # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
gpu        = '0'            # Used GPU
wght_decay = 0.00001        # Weight Decay
########################################

######### Paths  ##########
# Working station 
dataset_path = "E:\dataset\cifar-10-python"
save_path = "E:\ilex\"
###########################

### Initialization of some variables ###
class_means = np.zeros((512, nb_groups * nb_cl, 2, nb_groups))
loss_bathc = []
files_protoset = []
for _ in range(nb_groups * nb_cl):
    files_protoset.append([])

### Preparing the files for the training/validation ###

# Random mixing
print("Mixing the classes and putting them in batches of classes...")
np.random.seed(1993)
order = np.arange(nb_groups * nb_cl)
mixing = np.arrange(nb_groups * nb_cl)
np.random.shuffle(mixing)

# setting labels
all_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
labels_dic = {}

# Preparing the files per group of classes
print("Creating a validation set ...")
files_train, files_valid = utils_data.prepare_files(train_path, mixing, order, labels_dic, nb_groups, nb_cl, nb_val)

# with