#This file containes all the functions we have defined and used.

import numpy as np
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder
from random import choice, sample
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import fetch_openml
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_openml
import numpy as np

#This function takes an array indicating clean data with a value of 1, an array of labels, and data values, then provides information about the detected noisy data instances, including their indices, labels, and data.
def ns_detect(loss_record_mat_ns, y_train_ns, data_value, epoch,N_features):
    ns_data_index = []
    ns_label = []
    ns_output = list(loss_record_mat_ns[:,epoch])
    ns_data_index_temp = get_index(ns_output, 0.0)
    for z in range(len(ns_data_index_temp)):
         if (ns_data_index_temp[z] not in ns_data_index):
            ns_data_index.append(ns_data_index_temp[z])
    ns_data = np.zeros(shape=(len(ns_data_index), N_features), dtype=float)

    ns_label = []
    for j in range(len(ns_data_index)):
        da= ns_data_index[j]
        ns_data[j] = data_value[da,: ]
        ns_label.append(y_train_ns[da])
    return ns_data_index, ns_label, ns_data

# This function processes noisy data instances by receiving their indices, original labels, and post-noise labels, along with a list of k counterfactuals for each class. It outputs new labels for each noisy instance, counts of accurately detected labels, lists of correctly and incorrectly changed labels, and specifies noisy indices and labels for the next iteration.
def cf_search_correct_label(cf_correct_data, ns_data_index,myDevice, y_train):

    cf_loss = []
    loss_value_noise = torch.Tensor([0])           
    y_train_ns_next = y_train
    count = 0  
    cf_label = []
    cf_change_label = []
    
    for z in range(len(cf_correct_data)):
        cf_data_each = cf_correct_data[z]
        data_index_value = ns_data_index[z]
        dis_value_dict = {}  
        for key, value in cf_data_each.items():
            dis_value = value["dis_track"]
            if(len(dis_value) == 0):
                dis_value_dict.setdefault(key, 100)
            else:
                dis_value_dict.setdefault(key, dis_value[-1].numpy())
        find_label = min(dis_value_dict,key=dis_value_dict.get)
        cf_label.append(find_label)  
        
        loss_value = cf_data_each[find_label]["loss"]
        y_train_ns_next[data_index_value] = find_label
        loss_value_noise = loss_value_noise.to(myDevice) + loss_value.to(myDevice)

    return cf_label, y_train_ns_next, loss_value_noise

# This function receives the list of indices, simple loss for each data instance, list of unique labels and the predicted labels and outputs the peer loss value for each data instance.       
def peer_loss(list_of_indexes, ori_loss_list,list_of_unique_labels,predict_label_list,myDevice):
    all_loss_list=[]
    for m in list_of_indexes:### number of batch size
        ori_loss = ori_loss_list[m]
        myLoss=torch.nn.CrossEntropyLoss()
        thr_inital = torch.Tensor([0])
        for j in range(len(list_of_unique_labels)):
            logits = predict_label_list[m].unsqueeze(0).to(myDevice)
            target_value = list_of_unique_labels[j]
            target = torch.tensor([target_value], dtype=torch.long).to(myDevice)                           
            temp_value = myLoss(logits.reshape([1,-1]), torch.tensor([target], dtype=torch.long).to(myDevice))
            thr_inital = thr_inital.to(myDevice) + temp_value.to(myDevice)
        ave_loss = thr_inital/len(list_of_indexes)
        all_loss = ori_loss - ave_loss # this is peer loss value
        all_loss_list.append(all_loss)
    return all_loss_list

# This function receives the pretrained model and outputs the first threshold for noise detection component
def noise_threshold_selection(model,myDevice):
        model.eval()
        data_pretrain =torch.load("Dpre_clean.pth") 
        data_noisy=torch.load("Dpre_noisy.pth")
        features_list = []
        labels_list = []

        for feature, label in data_pretrain:
            features_list.append(feature)
            labels_list.append(label)

        features_tensor = torch.stack(features_list)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        myLoss = torch.nn.CrossEntropyLoss()
        lc_list_ori = []
        ln_list_ori = []
        ln2_list_ori=[]
        labels_list = []

        for _, label in data_pretrain :
            labels_list.append(label)           
        temp_label_list=labels_list.copy()
        def extract_labels(dataset):
            if isinstance(dataset, torch.utils.data.Subset):
                _, labels = dataset.dataset.tensors
                return labels[dataset.indices]
            else:
                _, labels = dataset.tensors
                return labels

        labels_pretrain = extract_labels(data_pretrain)
        labels_noisy = extract_labels(data_noisy)

        noisy_index_pre = (labels_pretrain != labels_noisy).nonzero(as_tuple=True)[0]

        ori_label_pre = labels_pretrain[noisy_index_pre]
        ns_label_pre = labels_noisy[noisy_index_pre]
        y_pre = labels_noisy
        #y_pre is the list of labels after introducing noise
                
        new_feature_tensor = features_tensor[:]
        new_dataset=TensorDataset(new_feature_tensor, torch.tensor(y_pre, dtype=torch.long))
        new_model = copy.deepcopy(model)
        new_model.train()
        train_loader = DataLoader(new_dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(50): #training g tilde
            for data, target in train_loader:
                optimizer.zero_grad()
                output = new_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        torch.save(new_model.state_dict(), 'UpdatedModel.pth')
        
        predict_label_list=[]
        predict_label_list2=[]
        for i, (image, label) in enumerate(data_pretrain):#label is real label
            image = image.to(myDevice)
            predict_label = model(image.unsqueeze(0))
            predict_label_list.append(predict_label)
        # Compute the loss
            lc_ori = myLoss(predict_label, torch.tensor([label], dtype=torch.long).to(myDevice))
            lc_list_ori.append(lc_ori.item()) 

            predict_label2=new_model(image.unsqueeze(0))
            predict_label_list2.append(predict_label2)
            ln_ori = myLoss(predict_label, torch.tensor([y_pre[i]], dtype=torch.long).to(myDevice))
            ln_list_ori.append(ln_ori.item())

            #here we want to calculate loss value of detected points in noisy distrubution
            ln2_ori = myLoss(predict_label2, torch.tensor([y_pre[i]], dtype=torch.long).to(myDevice))
            ln2_list_ori.append(ln2_ori.item())
            
        unique_labels = list(set(y_pre))    
        lc_list=peer_loss([i for i in range (len(y_pre))], lc_list_ori,unique_labels, predict_label_list, myDevice)
        ln_list=peer_loss([i for i in range (len(y_pre))], ln_list_ori,unique_labels, predict_label_list, myDevice)
        ln2_list=peer_loss([i for i in range (len(y_pre))], ln2_list_ori,unique_labels, predict_label_list2, myDevice)
        print("len lc_list",len(lc_list))
        print("len ln_list",len(ln_list))
  
        l_diff = [(lc - ln) for lc, ln in zip(lc_list, ln2_list)]
        clean_index_pre=[]
        
        for i in range(len(y_pre)):
            if i not in noisy_index_pre:
                clean_index_pre.append(i)
   
        noisy_distribution=[]
        print("len l_diff",len(l_diff))
        min_lost_pre, min_index = min((l_diff[j], j) for j in clean_index_pre)
        print("index of clean data with the least ldiff value", min_index)
        ave_clean_loss=(sum([l_diff[j] for j in clean_index_pre]))/len(clean_index_pre)
        print("min_lost_pre",min_lost_pre)
        for r in range(len(y_pre)):
            if l_diff[r]<=min_lost_pre:
                noisy_distribution.append(r)
        
        # Convert list values to x, and their indices to y
        x_values = l_diff  # Assuming l_diff is a list of tensors
        y_indices = list(range(len(l_diff)))

        # Convert all tensors in l_diff to a NumPy array
        x_values_np = np.array([x.detach().numpy() for x in x_values])

        # Initial scatter plot
        plt.scatter(x_values_np, y_indices, color='black')

        for index in noisy_index_pre:
            plt.scatter(x_values_np[index], y_indices[index], color='orange')
        for index in clean_index_pre:
            plt.scatter(x_values_np[index], y_indices[index], color='blue')

        min_lost_pre_value = min_lost_pre.item()  # Detaches and converts to Python scalar
        plt.axvline(x=min_lost_pre_value, color='green', linestyle='--', label=f'x = {min_lost_pre_value}')

        plt.xlabel('ldiff')
        plt.ylabel('Data index')
        plt.legend()
        plt.show()
        
        if len(noisy_distribution) > 0:
            sum_of_loss=0
            for m in noisy_distribution:
                sum_of_loss += ln_list[m]
            first_threshold = sum_of_loss / len(noisy_distribution)
            first_threshold=first_threshold.item()
        else:

            first_threshold = 0
        return first_threshold,lc_list,y_pre,data_pretrain,clean_index_pre, noisy_index_pre,ln2_list,new_dataset

# This function calculates the new noise threshold after the first iteration. It utilizes the trained model, the initially calculated loss for clean data instances, a list of labels, pretrain data, indices of clean data points, indices of noisy data instances, and the noisy pretrained dataset.

def update_threshold(model, last_lc_list,y_pre,data_pretrain,myDevice,clean_index_pre,noisy_index_pre,ln2_list,new_dataset):

        labels_list=[]
        ln_list_ori=[]
        ln2_list_ori=[]
        ln_for_thresh_list_ori=[]
        for _, label in data_pretrain :
            labels_list.append(label)
        lc_list_ori=[]
        predict_label_list=[]
        predict_label_list2=[]
    
        myLoss = torch.nn.CrossEntropyLoss()
        model.eval() 
        new_model = copy.deepcopy(model)
        new_model.train()
        train_loader = DataLoader(new_dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  
        criterion = nn.CrossEntropyLoss()

        for epoch in range(50): #training g tilde
            for data, target in train_loader:
                optimizer.zero_grad()
                output = new_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
        for i, (image, label) in enumerate(data_pretrain):
 
            image = image.to(myDevice)
            predict_label = model(image.unsqueeze(0))
            predict_label_list.append(predict_label)

            lc_ori = myLoss(predict_label, torch.tensor([label], dtype=torch.long).to(myDevice))
            lc_list_ori.append(lc_ori.item()) 
            
            predict_label2=new_model(image.unsqueeze(0))
            predict_label_list2.append(predict_label2)
            ln_ori = myLoss(predict_label2, torch.tensor([y_pre[i]], dtype=torch.long).to(myDevice))

            ln_list_ori.append(ln_ori.item())
            ln_for_thresh_ori= myLoss(predict_label, torch.tensor([y_pre[i]], dtype=torch.long).to(myDevice))
            ln_for_thresh_list_ori.append(ln_ori.item())

        list_of_unique_labels= list(set(y_pre))                                                                                       
        lc_list=peer_loss([i for i in range (len(y_pre))], lc_list_ori,list_of_unique_labels,predict_label_list,myDevice)
        ln_list=peer_loss([i for i in range (len(y_pre))], ln_list_ori,list_of_unique_labels,predict_label_list2,myDevice)
        ln_for_thresh=peer_loss([i for i in range (len(y_pre))], ln_for_thresh_list_ori,list_of_unique_labels,predict_label_list,myDevice)   
        l_diff = [(lc - ln) for lc, ln in zip(lc_list, ln_list)]        
        noisy_distribution=[]
    
        min_lost_pre = min([l_diff[j] for j in clean_index_pre])
        ave_clean_loss=(sum([l_diff[j] for j in clean_index_pre]))/len(clean_index_pre)
        
        x_values = l_diff 
        y_indices = list(range(len(l_diff)))

        # Convert all tensors in l_diff to a NumPy array
        x_values_np = np.array([x.detach().numpy() for x in x_values])

        # Initial scatter plot
        plt.scatter(x_values_np, y_indices, color='black')

        for index in noisy_index_pre:
            plt.scatter(x_values_np[index], y_indices[index], color='orange')
        for index in clean_index_pre:
            plt.scatter(x_values_np[index], y_indices[index], color='blue')

        min_lost_pre_value = min_lost_pre.item()  # Detaches and converts to Python scalar
        plt.axvline(x=min_lost_pre_value, color='green', linestyle='--', label=f'x = {min_lost_pre_value}')

        plt.xlabel('ldiff')
        plt.ylabel('Data index')
        plt.legend()
        plt.show()
        
        for r in range(len(y_pre)):
            if l_diff[r]<=min_lost_pre:
                noisy_distribution.append(r)
        if len(noisy_distribution) > 0:
            
            sum_of_loss=0
            for m in noisy_distribution:### number of batch size
                sum_of_loss += ln_for_thresh[m]

            first_threshold = sum_of_loss / len(noisy_distribution)
        else:
    # Handles the case where noisy_distribution is empty
            first_threshold = 0
        return first_threshold





