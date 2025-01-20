from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from skimage.io import imread
import torch
import scipy.io
import numpy as np
from scripts.config import BATCH_SIZE, PRE_TRAIN_EPOCHS, MODEL_FILENAME, LR, WORKERS, LATENT_DIM, NUM_CLASSES, LAMBDA, GAMMA, FINE_TUNE_EPOCHS, dataset_name , CHANNELS, TOLERANCE, UPDATE_INTERVAL, IHMC


def get_resized_transform():
    transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
    return transform
#*****************************************************************************

def get_normalized_transform():
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
    return transform
#*****************************************************************************

def get_simple_transform():
    transform = transforms.Compose([
      transforms.ToTensor()
  ])
    return transform
#*****************************************************************************

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256, num_workers=10):
    # Define the transforms
    # transform = get_resized_transform()
    transform = get_simple_transform()
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    train_dataset = datasets.CIFAR10('./data', train=True, download=download,
                                    transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)

    test_dataset = datasets.CIFAR10('./data', train=False, download=download, 
                                    transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading the MVP-N dataset    
#***************************************************************************** 
def get_MVPN_dataloaders(batch_size=256, num_workers=8):
    """MVP-N dataloader with (64x64) images."""
    transform = get_simple_transform()
    train_dataset = MVPN_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/MVP-N_Triplet_Train/', 
                              transform = transform)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    test_dataset = MVPN_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/MVP-N_Triplet_Test/', 
                              transform = transform)
    
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#**************************************************************
#--- Create a dataset of MVP-N data
#**************************************************************
class MVPN_Dataset(Dataset):
    """Dataset made with MVP-N 64x64 images"""          
    def __init__(self, csv_file, root_dir, transform=None):
          self.annotations = pd.read_csv(root_dir+csv_file)
          self.root_dir = root_dir
          self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        positive = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        negative = imread(img_path)
        target = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative, target, index
#**************************************************************


#*****************************************************************************
#--- Loading the Multi_Market dataset    
#***************************************************************************** 
def get_Multi_Market_dataloaders(batch_size=256, num_workers=8):
    """Multi_Market dataloader with (64x, 128) images."""
    transform = get_normalized_transform()
    train_dataset = Multi_Market_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi-Market_Triplet_Train/', 
                              transform = transform)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    test_dataset = Multi_Market_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi-Market_Triplet_Test/', 
                              transform = transform)
    
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#**************************************************************
#--- Create a dataset of Multi_Market data
#**************************************************************
class Multi_Market_Dataset(Dataset):
    """Dataset made with Multi-Market 64 x 128 images"""          
    def __init__(self, csv_file, root_dir, transform=None):
          self.annotations = pd.read_csv(root_dir+csv_file)
          self.root_dir = root_dir
          self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        positive = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        negative = imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative, y_label, index
#**************************************************************

#*****************************************************************************
#--- Loading the Multi_MNIST dataset    
#***************************************************************************** 
def get_Multi_MNIST_dataloaders(batch_size=256, num_workers=8):
    """Multi_MNIST dataloader with (32, 32) images."""    
    transform = get_simple_transform()  
    train_dataset = Multi_MNIST_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi MNIST/', 
                              transform = transform)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    test_dataset = Multi_MNIST_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi MNIST/', 
                              transform = transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#**************************************************************
#--- Create a dataset of Multi_MNIST data
#**************************************************************
class Multi_MNIST_Dataset(Dataset):
    """Dataset made with Multi-MNIST 32 x 32 images"""          
    def __init__(self, csv_file, root_dir, transform=None):
          self.annotations = pd.read_csv(root_dir+csv_file)
          self.root_dir = root_dir
          self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        positive = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        negative = imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative, y_label, index
#**************************************************************

#*****************************************************************************
#--- Loading the Multi_Fashion dataset    
#***************************************************************************** 
def get_Multi_FASHION_dataloaders(batch_size=256, num_workers=8):
    """Multi_Fashion dataloader with (32, 32) images."""   
    transform = get_simple_transform()
    if IHMC:                                                                    
        train_dataset = Multi_FASHION_Dataset(csv_file = 'data.csv', root_dir = '/workspace/Datasets/Multi_Fashion_Test/', 
                              transform = transform)
    else:                  
        train_dataset = Multi_FASHION_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi_Fashion_Test/', 
                              transform = transform)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    if IHMC:
        test_dataset = Multi_FASHION_Dataset(csv_file = 'data.csv', root_dir = '/workspace/Datasets/Multi_Fashion_Test/', 
                              transform = transform)
    else:
        test_dataset = Multi_FASHION_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi_Fashion_Test/', 
                              transform = transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#**************************************************************
#--- Create a dataset of Multi_Fashion data
#**************************************************************
class Multi_FASHION_Dataset(Dataset):
    """Dataset made with Multi-Fashion 32 x 32 images"""          
    def __init__(self, csv_file, root_dir, transform=None):
          self.annotations = pd.read_csv(root_dir+csv_file)
          self.root_dir = root_dir
          self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        positive = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        negative = imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative, y_label, index
#**************************************************************

#*****************************************************************************
#--- Loading the Multi_STL_10 dataset    
#***************************************************************************** 
def get_Multi_STL_10_dataloaders(batch_size=256, num_workers=8):
    """Multi_STL dataloader with (64, 64) images."""   
    transform = get_simple_transform()                  
    train_dataset = Multi_STL_10_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi-STL_10_Train/', 
                              transform = transform)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    test_dataset = Multi_STL_10_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi-STL_10_Train/', 
                              transform = transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#**************************************************************
#--- Create a dataset of Multi_STL_10 data
#**************************************************************
class Multi_STL_10_Dataset(Dataset):
    """Dataset made with Multi-STL_10 64 x 64 images"""          
    def __init__(self, csv_file, root_dir, transform=None):
          self.annotations = pd.read_csv(root_dir+csv_file)
          self.root_dir = root_dir
          self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        positive = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        negative = imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative, y_label, index
#**************************************************************


#--- For load .mat files
class MULTI_MNIST(Dataset):  
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MULTI_MNIST.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MULTI_MNIST.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MULTI_MNIST.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MULTI_MNIST.mat')['X3'].astype(np.float32)
        
    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x1 = x1.transpose(2,0,1) #Reorder to be in the format wanted (1, 32, 32)
        x2 = self.V2[idx]
        x2 = x2.transpose(2,0,1) 
        x3 = self.V3[idx]
        x3 = x3.transpose(2,0,1) 
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MULTI_FASHION(Dataset):  
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MULTI_FASHION.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MULTI_FASHION.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MULTI_FASHION.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MULTI_FASHION.mat')['X3'].astype(np.float32)
        
    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x1 = x1.transpose(2,0,1) #Reorder to be in the format wanted (1, 32, 32)
        x2 = self.V2[idx]
        x2 = x2.transpose(2,0,1) 
        x3 = self.V3[idx]
        x3 = x3.transpose(2,0,1) 
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MULTI_MVP_N(Dataset):  
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MULTI_MVP-N.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MULTI_MVP-N.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MULTI_MVP-N.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MULTI_MVP-N.mat')['X3'].astype(np.float32)
        
    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x1 = x1.transpose(2,0,1) #Reorder to be in the format wanted (3, 64, 64)
        x2 = self.V2[idx]
        x2 = x2.transpose(2,0,1) 
        x3 = self.V3[idx]
        x3 = x3.transpose(2,0,1) 
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MULTI_STL_10(Dataset):  
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['X3'].astype(np.float32)
        
    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x1 = x1.transpose(2,0,1) #Reorder to be in the format wanted (3, 96, 96)
        x2 = self.V2[idx]
        x2 = x2.transpose(2,0,1) 
        x3 = self.V3[idx]
        x3 = x3.transpose(2,0,1) 
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    cwd = os.getcwd()
    if dataset == "MULTI-MNIST": 
        dataset = MULTI_MNIST('./data/')
        dims = [1024, 1024]
        view = 2
        class_num = 10
        data_size = 5000    
    elif dataset == "MULTI-FASHION": 
        dataset = MULTI_FASHION('./data/')
        dims = [1024, 1024]
        view = 2
        class_num = 10
        data_size = 5000   
    elif dataset == "MULTI-MVP-N": 
        dataset = MULTI_MVP_N('./data/')
        dims = [12288, 12288]
        view = 2
        class_num = 17
        data_size = 5000  
    elif dataset == "MULTI-STL-10": 
        dataset = MULTI_STL_10('./data/')
        dims = [27648, 27648]
        view = 2
        class_num = 10
        data_size = 5000          
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num






