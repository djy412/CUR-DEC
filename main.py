# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:08:42 2024
@author: djy41
"""
from scripts.config import BATCH_SIZE, PRE_TRAIN_EPOCHS, MODEL_FILENAME, LR, WORKERS, LATENT_DIM, NUM_CLASSES, LAMBDA, GAMMA, FINE_TUNE_EPOCHS, dataset_name , CHANNELS, TOLERANCE, UPDATE_INTERVAL
from classes.FC_Classifier import FC_NN
from Visualizations.Visualization import Show_settings, Show_dataloader_data, Show_Training_Loss, Show_Component_Embeddings, Show_Componet_Reconstructions, Show_Embedding_Space, Show_Complete_Reconstructions, Show_Partial_Embedding_Space, Show_Results, Show_Representation, Show_NMI_By_Epochs, Show_Variance
from scripts.data_loading import load_data, get_MVPN_dataloaders, get_Multi_Market_dataloaders, Multi_Market_Dataset, get_Multi_MNIST_dataloaders, Multi_MNIST_Dataset, get_MVPN_dataloaders, MVPN_Dataset, get_Multi_FASHION_dataloaders, Multi_FASHION_Dataset, get_Multi_STL_10_dataloaders, Multi_STL_10_Dataset
from scripts.utils import train_epoch, test_epoch, plot_ae_outputs, cluster_acc, calculate_purity, set_seed 
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from torch.optim import Adam
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from skimage.metrics import structural_similarity as ssim
import itertools
import seaborn as sns
import pandas as pd
import os

PreTRAIN = True
SEED = 1

set_seed(SEED)

#************************************************************************
#--- Define Convolutional Auto Encoder
#************************************************************************
class CAE(nn.Module):
    def __init__(self, LATENT_DIM):
        super(CAE, self).__init__()
        # Encoder_Common
        self.encoder_c = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 32, 64, 64)
            nn.Conv2d(32, 64, 3, padding=1),  # (batch_size, 64, 64, 64)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 64, 32, 32)
            nn.Conv2d(64, 128, 3, padding=1),  # (batch_size, 128, 32, 32)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 128, 16, 16)
            nn.Conv2d(128, 256, 3, padding=1),  # (batch_size, 256, 16, 16)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 256, 8, 8)
            
            #--- use for 96x96 or larger
            #nn.Conv2d(256, 512, 3, padding=1),  # (batch_size, 512, 16, 16)
            #nn.ReLU(True),
            #nn.MaxPool2d(2, 2),  # (batch_size, 512, 8, 8)
                   
            nn.Flatten(),
            #nn.Linear(512*3*3, LATENT_DIM) #--- 96x96 images
            #nn.Linear(256*6*6, LATENT_DIM) #---  96x96 images
            #nn.Linear(256*4*4, LATENT_DIM) #--- 64x64 images
            nn.Linear(256*2*2, LATENT_DIM) #--- 32x32 images
            
        )
        self.encoder_p = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 32, 64, 64)
            nn.Conv2d(32, 64, 3, padding=1),  # (batch_size, 64, 64, 64)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 64, 32, 32)
            nn.Conv2d(64, 128, 3, padding=1),  # (batch_size, 128, 32, 32)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 128, 16, 16)
            nn.Conv2d(128, 256, 3, padding=1),  # (batch_size, 256, 16, 16)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 256, 8, 8)

            #--- use for 96x96 or larger
            #nn.Conv2d(256, 512, 3, padding=1),  # (batch_size, 512, 16, 16)
            #nn.ReLU(True),
            #nn.MaxPool2d(2, 2),  # (batch_size, 512, 8, 8)
                        
            nn.Flatten(),
            #nn.Linear(512*3*3, LATENT_DIM) #--- 96x96 images
            #nn.Linear(256*6*6, LATENT_DIM) #---  96x96 images
            #nn.Linear(256*4*4, LATENT_DIM) #---  64x64 images
            nn.Linear(256*2*2, LATENT_DIM) #--- 32x32 images
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM*2, 256*2*2), #--- 32x32 images
            #nn.Linear(LATENT_DIM*2, 256*4*4), #--- this is for 64x64 images
            #nn.Linear(LATENT_DIM*2, 256*6*6), #--- this is for 96x96 images
            #nn.Linear(LATENT_DIM*2, 512*3*3), #--- this is for 96x96 images
            
            nn.Unflatten(1, (256, 2, 2)), #--- 32x32 images
            #nn.Unflatten(1, (512, 4, 2)), #---  
            #nn.Unflatten(1, (256, 4, 4)), #--- 64x64 images
            #nn.Unflatten(1, (256, 6, 6)), #--- 96x96 images
            #nn.Unflatten(1, (512, 3, 3)), #--- 96x96 images
 
            #--- use for 96x96 or larger
            #nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            #nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, CHANNELS, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            #nn.Tanh()
        )

    def pretrain(self, data_loader):
        pretrain_ae(self, data_loader)

    def forward(self, x):
        c = self.encoder_c(x)
        u = self.encoder_p(x) 
        z = torch.concat([c, u], dim=1)
        x_bar = self.decoder(z)
        return x_bar, c, u
# ----------------------------------------------------------------------------


#************************************************************************
#--- Define Common and Unique Model
#************************************************************************
class C_U_Model(nn.Module):
    def __init__(self, LATENT_DIM):
        
        super(C_U_Model, self).__init__()
        self.alpha = 1.0
        self.LATENT_DIM = LATENT_DIM
        self.ae = CAE(LATENT_DIM)
        
        #--- cluster layer
        self.cluster_layer = Parameter(torch.Tensor(NUM_CLASSES, LATENT_DIM))
        torch.nn.init.xavier_normal_(self.cluster_layer.data) #--- Initialize weight from normal distribution

    def pretrain(self, data_loader):
        pretrain_ae(self.ae, data_loader)

    def forward(self, a, p):
        if p == None: #--- Just use single view
            _, _, u_a = self.ae(a)
            #--- Create z_k based off unique and centers 
            z_a_u = u_a.repeat_interleave(NUM_CLASSES, dim = 0) #--- Shape (2560, 10) repeating u NUM_class times
            z_c = self.cluster_layer.repeat(u_a.size(0),1)
                
            z_a_k = torch.cat((z_c, z_a_u), dim = 1)
                
            #--- test the error of each peculiar image and centers to the input image
            split_a_z_k = torch.split(z_a_k, 10)
                
            mse_errs = torch.zeros(u_a.size(0), NUM_CLASSES)
            mse_errs_a = torch.zeros(u_a.size(0), NUM_CLASSES)
            
            for i in range(u_a.size(0)): 
                a_hat_k = self.ae.decoder(split_a_z_k[i])  
                for j in range(NUM_CLASSES):
                    squared_diff_a = (a_hat_k[j] - a[i])**2  
                    mse_errs_a[i][j] =  squared_diff_a.mean() #--- size(256, 10)
     
            mse_errs = mse_errs_a
            _, cluster_est = torch.min(mse_errs, dim = 1) #--- size(256)
            
        else:
            _, _, u_a = self.ae(a)
            _, _, u_p = self.ae(p)
            #--- Create z_k based off unique and centers 
            z_a_u = u_a.repeat_interleave(NUM_CLASSES, dim = 0) #--- Shape (2560, LATENT_DIM) repeating u NUM_class times
            z_p_u = u_p.repeat_interleave(NUM_CLASSES, dim = 0) #--- Shape (2560, LATENT_DIM) repeating u NUM_class times
            z_c = self.cluster_layer.repeat(u_a.size(0),1)
                
            z_a_k = torch.cat((z_c, z_a_u), dim = 1)
            z_p_k = torch.cat((z_c, z_p_u), dim = 1)
                
            #--- test the error of each peculiar image and centers to the input image
            split_a_z_k = torch.split(z_a_k, NUM_CLASSES)
            split_p_z_k = torch.split(z_p_k, NUM_CLASSES)
            #--- Initalize error to zero    
            mse_errs = torch.zeros(u_a.size(0), NUM_CLASSES)
            mse_errs_a = torch.zeros(u_a.size(0), NUM_CLASSES)
            mse_errs_p = torch.zeros(u_a.size(0), NUM_CLASSES)
            
            for i in range(u_a.size(0)): 
                a_hat_k = self.ae.decoder(split_a_z_k[i])  
                p_hat_k = self.ae.decoder(split_p_z_k[i])  
                for j in range(NUM_CLASSES): #--- Compare input with reconstructed unique and cluster centers images
                    squared_diff_a = (a_hat_k[j] - a[i])**2  
                    squared_diff_p = (p_hat_k[j] - p[i])**2  
                    mse_errs_a[i][j] =  squared_diff_a.mean() #--- take the mean of the image squared difference
                    mse_errs_p[i][j] =  squared_diff_p.mean()     
            
            #--- Now normalize the mse of the batch
            combined_min = torch.min(mse_errs_a.min(), mse_errs_p.min())
            combined_max = torch.max(mse_errs_a.max(), mse_errs_p.max())
            mse_errs_a = (mse_errs_a - combined_min) / (combined_max - combined_min)
            mse_errs_p = (mse_errs_p - combined_min) / (combined_max - combined_min)

            mse_errs = torch.min(mse_errs_a, mse_errs_p)  
         
            _  , cluster_est = torch.min(mse_errs, dim = 1) #--- size(256)

        return cluster_est
#----------------------------------------------------------------------------


#************************************************************************
#--- Define Triplet Loss
#************************************************************************
class Triplet_Loss_with_Reconstruction(nn.Module):
    def __init__(self, margin=1):
        super(Triplet_Loss_with_Reconstruction, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative, decoded_a_batch, decoded_p_batch, decoded_n_batch, latent_a_batchC, latent_p_batchC, latent_n_batchC):
        d2 = torch.norm(latent_a_batchC - latent_p_batchC, p=2, dim=1)
        d1 = torch.norm(latent_a_batchC - latent_n_batchC, p=2, dim=1)
            
        loss1 = F.mse_loss(decoded_a_batch, anchor) + F.mse_loss(decoded_p_batch, positive) +F.mse_loss(decoded_n_batch, negative) 
        loss2 = torch.mean(torch.relu(d2 - d1 + self.margin)) #--- This pushes common embeddings together 

        return loss1 + LAMBDA*loss2
#----------------------------------------------------------------------------

#************************************************************************
#--- Define updated Cluster Loss
#************************************************************************
class Cluster_loss(nn.Module):
    def __init__(self, margin = 1):
        super(Cluster_loss, self).__init__()
        self.margin = margin
        
    def forward(self, confidence, z_center, z_a, z_p, z_n, hat_a, a, hat_p, p, hat_n, n):
        d1 = torch.norm(z_center - z_a, p=2, dim=1) #--- Push anchor common embeddings to the cluster center
        d2 = torch.norm(z_center - z_n, p=2, dim=1) 
        loss1 = torch.mean(torch.relu(d1 - d2 + self.margin)) #--- Push the negative to the margin
        
        d3 = torch.norm(z_center - z_p, p=2, dim=1) #--- This pushes positive common embeddings to the cluster center
        loss2 = torch.mean(d3) 
        
        loss3 = F.mse_loss(hat_a, a) + F.mse_loss(hat_p, p)  + F.mse_loss(hat_n, n)

        return GAMMA*loss1 + GAMMA*loss2 + loss3
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Triplet Training
#************************************************************************
def pretrain_ae(model, data_loader):

    if PreTRAIN == True:
        #--- Create optimizer
        optimizer = Adam(model.parameters(), LR)
        
        #---Define loss function
        loss_fn = Triplet_Loss_with_Reconstruction()
        
        Loss_histroy = [0]
        for epoch in range(PRE_TRAIN_EPOCHS):
            total_loss = 0.0

            for batch_idx, (a, p, n, labels, _) in enumerate(data_loader):
                a = a.to(device)
                p = p.to(device)
                n = n.to(device)
    
                optimizer.zero_grad()
                a_hat, latent_a_batchC, latent_a_batchP = model(a)
                p_hat, latent_p_batchC, latent_p_batchP = model(p)
                n_hat, latent_n_batchC, latent_n_batchP = model(n)
                loss = loss_fn(a, p, n, a_hat, p_hat, n_hat, latent_a_batchC, latent_p_batchC, latent_n_batchC)
                    
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                Loss_histroy.append(loss.item())
            print("epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx + 1)))
        
        #--- Save the model weights 
        if dataset_name == 'Dataset: Multi-MNIST':
            torch.save(model.state_dict(), 'data/Final/MNIST_Triplet_ae.pkl')
        if dataset_name == 'Dataset: Multi-FASHION':
            torch.save(model.state_dict(), 'data/Final/FASHION_Triplet_ae.pkl')
        if dataset_name == 'Dataset: Multi-Market':
            torch.save(model.state_dict(), 'data/Final/Market_Triplet_ae.pkl')
        if dataset_name == 'Dataset: Multi-MVP-N':
            torch.save(model.state_dict(), 'data/Final/MVP-N_Triplet_ae.pkl')
        if dataset_name == 'Dataset: Multi-STL-10':
            torch.save(model.state_dict(), 'data/Final/STL-10_Triplet_ae.pkl')
        if dataset_name == 'MULTI-MNIST':
            torch.save(model.state_dict(), 'data/Final/Multi-MNIST_Triplet_ae.pkl')
        if dataset_name == 'MULTI-FASHION':
            torch.save(model.state_dict(), 'data/Final/Multi-FASHION_Triplet_ae.pkl')
        if dataset_name == 'MULTI-MVP-N':
            torch.save(model.state_dict(), 'data/Final/Multi-MVP-N_Triplet_ae.pkl') 
        if dataset_name == 'MULTI_STL-10':
            torch.save(model.state_dict(), 'data/Final/Multi-STL-10_Triplet_ae.pkl')             
            
        print("model saved to data/Final/'Dataset_name' Triplet_ae.pkl")

    else:
        if dataset_name == 'Dataset: Multi-MNIST':    
            load_model_path = 'data/Final/MNIST_Triplet_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-FASHION':
            load_model_path = 'data/Final/FASHION_Triplet_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-Market':
            load_model_path = 'data/Final/Market_Triplet_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-MVP-N':
            load_model_path = 'data/Final/MVP-N_Triplet_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'Dataset: Multi-STL-10':  
            load_model_path = 'data/Final/STL-10_Triplet_ae.pkl'    
            model.load_state_dict(torch.load(load_model_path))         
        if dataset_name == 'MULTI-MNIST':    
             load_model_path = 'data/Final/Multi-MNIST_Triplet_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path))     
        if dataset_name == 'MULTI-FASHION':    
             load_model_path = 'data/Final/Multi-FASHION_Triplet_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'MULTI-MVP-N':  
             load_model_path = 'data/Final/Multi-MVP-N_Triplet_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        if dataset_name == 'MULTI_STL-10':  
             load_model_path = 'data/Final/Multi-STL-10_Triplet_ae.pkl'    
             model.load_state_dict(torch.load(load_model_path)) 
        for batch_idx, (a, _, _, labels, _) in enumerate(data_loader):
            with torch.no_grad():
                a = a.to(device)
                a_hat, latent_a_batchC, latent_a_batchP = model(a)
            break   
#----------------------------------------------------------------------------


#*****************************************************************************
#--- Main Function
#*****************************************************************************
if __name__=='__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
       
    print('Loading data...')
    if dataset_name == 'Dataset: Multi-MVP-N':
        train_loader, test_loader, _, _ = get_MVPN_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    elif dataset_name == 'Dataset: Multi-FASHION':
        train_loader, test_loader, _, _ = get_Multi_FASHION_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    elif dataset_name == 'Dataset: Multi-Market':
        train_loader, test_loader, _, _ = get_Multi_Market_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    elif dataset_name == 'Dataset: Multi-MNIST':
        train_loader, test_loader, _, _ = get_Multi_MNIST_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    elif dataset_name == 'Dataset: Multi-STL-10':
        train_loader, test_loader, _, _ = get_Multi_STL_10_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    elif dataset_name == 'MULTI-MNIST':
        dataset, dims, view, data_size, class_num = load_data("MULTI-MNIST")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-FASHION':
        dataset, dims, view, data_size, class_num = load_data("MULTI-FASHION")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI-MVP-N':
        dataset, dims, view, data_size, class_num = load_data("MULTI-MVP-N")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
    elif dataset_name == 'MULTI_STL-10':
        dataset, dims, view, data_size, class_num = load_data("MULTI-STL-10")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,)
 
    #--- Define the Common and Peculiar Model
    model = C_U_Model(LATENT_DIM).to(device)

    #--- Train the model with Triplet
    model.pretrain(test_loader)             

    #--- Create optimizer
    optimizer1 = Adam(model.parameters(), LR)

    #--- Initialize cluster centers 
    Full_data_a, Full_data_p, y_true=[], [], []
    
    for a, p, _, y, _ in test_loader:
        Full_data_a.append(a)
        Full_data_p.append(p)
        y_true.append(y)

    Full_data_a = torch.cat(Full_data_a).to(device)
    Full_data_p = torch.cat(Full_data_p).to(device)
    y_true = torch.cat(y_true)
    
    # Clear unused data
    a = p = None
    torch.cuda.empty_cache()

    with torch.no_grad():
        _, c, u = model.ae(Full_data_a)   
        
    #--- Cluster based on only the common representation, find the cluster centers using
    #--- k-means THEN find the closets representation to that center as the ACTUAL center
    kmeans = KMeans(n_clusters=NUM_CLASSES, n_init=20)
    y_pred = kmeans.fit_predict(c.cpu().numpy())
    nmi_k = nmi_score(y_pred, y_true)
    print("Start Kmeans nmi score={:.4f}".format(nmi_k))
    
    #--- kmeans.cluster_centers_ has the centers, now cycle through the anchor images to find
    #--- the ones closets to the centers
    cluster_centers = []
    for k_center in kmeans.cluster_centers_:
        distances = []
        for anchor_latent in c:
            distance = torch.sqrt(torch.sum((anchor_latent.cpu() - k_center) ** 2))
            distances.append(distance)
        # Find the minimum index
        distances = torch.stack(distances)
        _, min_index = torch.min(distances, dim=0)
        cluster_centers.append(c[min_index])  
    cluster_centers = torch.stack(cluster_centers)    
    print("Cluster centers: ", cluster_centers.size())  
        
    #--- Load the initial cluster centers into model
    #model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.cluster_layer.data = cluster_centers.to(device)

    c = u = None    
    torch.cuda.empty_cache()
    
    # Semantic labels
    center_labels = torch.arange(NUM_CLASSES).unsqueeze(1)
    
    #--- Soft labels only based off the common representation
    with torch.no_grad():
        cluster_est = model(Full_data_a, Full_data_p)
    y_pred = cluster_est.cpu().numpy()
        
    acc = cluster_acc(y_true, y_pred, NUM_CLASSES)
    nmi = nmi_score(y_true, y_pred)
    pur = calculate_purity(y_true, y_pred)
    print('Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
  
    Start_ACC = acc
    Start_NMI = nmi
    Start_PUR = pur
    
    # Clear all unneeded data
    Full_data_a = Full_data_p = y_true = cluster_est = None
    torch.cuda.empty_cache()
    
    #---Define loss function
    loss_fn = Cluster_loss()
    NMI_histroy = []    
    model.train()

    #--- "Fine-tunning" training
    for epoch in range(FINE_TUNE_EPOCHS):
        total_loss = 0   
        #--- Calculate the cluster centers 
        if epoch % UPDATE_INTERVAL == 0:
            mu_est = [torch.zeros(5000, LATENT_DIM, device=device), torch.zeros(5000, device=device, dtype=torch.long)]
            est_confidence = [torch.zeros(5000, device=device)]
                    
            #--- Calculate new "truth" data based on shuffled data
            #--- don't need the data used, just the index estimates of the cluster center representations 
            y_pred_total, y_true = [],[]  # To collect y_pred y_true across all batches
            for a, p, _, y, idx in test_loader:  # Using shuffled data
                a = a.to(device)
                p = p.to(device)
                    
                cluster_est = model(a, p)  # Get the model estimation of cluster centers
                y_pred = cluster_est.cpu().numpy()
                y_pred_total.extend(y_pred)  # Collect y_pred for accuracy, NMI, ARI calculations
            
                # Ensure correct assignment by indexing correctly
                mu_est[0][idx] = model.cluster_layer.data[cluster_est].to(device)  # Get the est centers latent representations
                mu_est[1][idx] = idx.to(device).long()
                     
                # Collect y_true for accuracy, NMI, ARI calculations
                y_true.extend(y.cpu().numpy())
                    
            # Compute accuracy, NMI, ARI
            acc = cluster_acc(y_true, y_pred_total, NUM_CLASSES)
            nmi = nmi_score(y_true, y_pred_total)
            pur = calculate_purity(y_true, y_pred_total)
            print(f'Iter {epoch}: Acc {acc:.4f}, nmi {nmi:.4f}, purity {pur:.4f}')
            NMI_histroy.append(nmi)
                    
            a = p = y = idx = None  # Clear variables for the next iteration
           
        for i, (a, p, n, _, idx) in enumerate(test_loader):
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)
                    
            optimizer1.zero_grad()   
    
            a_hat, common_a, _ = model.ae(a)
            p_hat, common_p, _ = model.ae(p)
            n_hat, common_n, _ = model.ae(n)
        
            loss = loss_fn(1, mu_est[0][idx], common_a, common_p, common_n, a_hat, a, p_hat, p, n_hat, n)
                
            total_loss += loss.item()
            total_loss += loss.item()
        
            loss.backward()
            optimizer1.step()
        print("epoch {} loss={:.4f}".format(epoch, total_loss / (i + 1)))
            

    Full_data_a, Full_data_p, y_true=[], [], [] 
    torch.cuda.empty_cache()
    for a, p, _, y, _ in test_loader:
        Full_data_a.append(a)
        Full_data_p.append(p)
        y_true.append(y)

    Full_data_a = torch.cat(Full_data_a).to(device)
    Full_data_p = torch.cat(Full_data_p).to(device)
    y_true = torch.cat(y_true)
      
    #--- Show the full common feature embeddings using t-SNE 
    with torch.no_grad():
        a_bar, common, u = model.ae(Full_data_a)
            
    # cluster only based off the common representation
    with torch.no_grad():
        cluster_est = model(Full_data_a, Full_data_p)
    y_pred = cluster_est.numpy()
        
    acc = cluster_acc(y_true, y_pred, NUM_CLASSES)
    nmi = nmi_score(y_true, y_pred)
    pur = calculate_purity(y_true, y_pred)
    print('Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
    
    END_ACC = acc
    END_NMI = nmi
    END_PUR = pur
    Show_Results(SEED, Start_ACC, Start_NMI, Start_PUR, END_ACC,  END_NMI, END_PUR)
    
    y_pred = kmeans.fit_predict(common.cpu().numpy())
    nmi_k = nmi_score(y_pred, y_true)
    print("End Kmeans nmi score={:.4f}".format(nmi_k))