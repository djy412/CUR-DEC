# Parameter configurations
BATCH_SIZE = 124
PRE_TRAIN_EPOCHS = 500
FINE_TUNE_EPOCHS = 100
MODEL_FILENAME = './data/model.ckpt'
# Learning rate for both encoder and decoder
LR = 0.001
LAMBDA = 0.1 #--- How much reconstruction loss to include in Triplet loss
GAMMA = 0.5 #--- How much clustering factor to include in the Clutering loss
LATENT_DIM = 12
NUM_CLASSES = 10
TOLERANCE = 0.001 #--- How close to the last estimage is good enough
UPDATE_INTERVAL = 10 #--- How often to update the estimated "true data", 1 would = updating every Epoch
# Load data in parallel by choosing the best num of workers for your system
WORKERS = 8
#dataset_name = 'Dataset: Multi-MNIST'
dataset_name = 'MULTI-MNIST'
#dataset_name = 'Dataset: Multi-FASHION'
#dataset_name = 'MULTI-FASHION'
#dataset_name = 'Dataset: Multi-Market'
#dataset_name = 'Dataset: Multi-MVP-N'
#dataset_name = 'MULTI-MVP-N'
#dataset_name = 'Dataset: Multi-STL-10'
#dataset_name = 'MULTI_STL-10'
CHANNELS = 1
IHMC = False