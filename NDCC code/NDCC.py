#This file contains the main algorithm and implements noise detection, noise corection components and trains the model using the corrected dataset

from tqdm import tqdm
import functionn
from functionn import *
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import pickle
from torchmetrics.image.kid import KernelInceptionDistance
import copy
from torch.utils.data import TensorDataset
import torchvision.models as models
import sys
sys.path.append('/path/to/the/directory/containing/Pre_FF_fmodel')
from Pre_FF_fmodel import Net
import time

#To create the counterfactuals, it learns the optimal input (self.x) instead of weights. it will adjust the values of self.x along with any other learnable parameters in an attempt to minimize the loss
class RESNET_MF(nn.Module):  
    def __init__(self, X_ori):
        super(RESNET_MF, self).__init__()
        self.x = torch.nn.Parameter(X_ori, requires_grad=True) # this code makes it a learnable parameter
    def forward(self, model):#, t1, b1, t2, b2, t3, b3):
        self.o2 =model(self.x)
        return self.o2, self.x    

class NDCC():
    def __init__(self, input_data_size, train_batch_size, myModel, myDevice, train_dataset, learning_rate):
        self.train_dataset = train_dataset
        self.size = input_data_size
        self.nr = 0
        self.train_bs = train_batch_size
        self.myModel = myModel
        self.myDevice = myDevice
        self.noisy_index = None
        self.x = []
        self.y = []
        self.y_ori = []
        self.classes = []
        self.lr = learning_rate
        self.myModel.eval()
        self.myOptimzier_ns = optim.AdamW(myModel.parameters(), lr = learning_rate)
        
        # Correctly handle TensorDataset
        if isinstance(self.train_dataset, torch.utils.data.TensorDataset):
            self.features_tensor, self.labels_tensor = self.train_dataset.tensors
        else:
            raise ValueError("The provided dataset is not a TensorDataset.")
        
        for i in range(len(self.features_tensor)):
            data = self.features_tensor[i]
            label = self.labels_tensor[i]
            self.x.append(data)
            self.y.append(label)
            self.y_ori.append(label)  
        self.numFeatures=len(self.x[0])
    
    def train_ns_prepare(self, num_iteration, dataset):
        self.num_batch_iter = int(self.size/self.train_bs)
        num_data = self.num_batch_iter * self.train_bs
        self.classes = list(set(tensor.item() for tensor in self.y))
        self.data_sample = np.zeros(shape=(num_data, 70), dtype=float)

        for z in range(num_data):
            self.data_sample[z] = self.x[z]
          
        # 4 Lists for loss recordings
        self.loss_record_mat_ori = np.zeros(shape = (num_data, num_iteration)) ###keeps the loss of each data instance in each epoch in noise detection process
        self.loss_record_mat_all = np.zeros(shape = (num_data, num_iteration))
        self.loss_record_mat_ns = np.zeros(shape = (num_data, num_iteration))
        self.loss_record_mat_pre_label = np.zeros(shape = (num_data, num_iteration))
        features_list = []
        labels_list = []

        for features, label in self.train_dataset:
            features_list.append(features)
            labels_list.append(label)

        self.data_value_test = torch.stack(features_list)
        self.label_value_test = torch.stack(labels_list)
        self.train_dataset.targets = self.y
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.train_bs, shuffle=False, num_workers=0, drop_last= False)
            
        return self.y
    
   
    def training(self, num_iteration):
        Initial_threshold,last_lc_list,y_pre,data_pretrain,clean_index_pre,noisy_index_pre,ln2_list, new_dataset=\
        noise_threshold_selection(self.myModel,self.myDevice)
        print("Initial threshold is:", Initial_threshold)
        torch.save(self.myModel, 'FFNeuralNetwork')
        cf_load_model = torch.load("FFNeuralNetwork")
        cf_model=copy.deepcopy(cf_load_model)
        cf_model.train()
        prev_y_tensor= torch.tensor(self.y, dtype=torch.long)
        y_tensor= torch.tensor(self.y, dtype=torch.long)
        real_noisy_index=self.noisy_index
        current_noisy_index=self.noisy_index
        Xn=[]
        Xr=[]
        for tt in tqdm(range(num_iteration)):
            epoch =  tt 
            start_time = time.time()
            self.myModel.eval() 
            
        # Noise detection training process
            myLoss = torch.nn.CrossEntropyLoss() 
            training_loss = 0.0
            data_pre_num = 0
            loss_value_clean  = torch.Tensor([0])
            self.y = [float(label.item()) if torch.is_tensor(label) else float(label) for label in self.y]
            
            for _step, input_data in enumerate(self.train_loader):
                if (_step < self.num_batch_iter):
                    
                    image, label = input_data[0].to(self.myDevice), input_data[1].to(self.myDevice) 
                    data_pre_num  = _step * (len(label))
                    predict_label = cf_model.forward(image).to(self.myDevice) # It defines how the model processes input data.
                    label_get = list(predict_label.max(1))  #returns two values: for each data instance in the batch, it gives the highest logit/probability value and indices of these maximum values, which correspond to the predicted class labels
                    self.loss_record_mat_pre_label[data_pre_num: data_pre_num +len(label) ,tt] = label_get[1].numpy()
                    for m in range(len(label)):# number of batch size
                        ori_loss = myLoss(predict_label[m].reshape([1,-1]), torch.tensor([self.y[m]], dtype=torch.long).to(self.myDevice))
                       ##loss for each data point in the batch
                        self.loss_record_mat_ori[m + data_pre_num, epoch] = ori_loss.detach() 
                        thr_inital = torch.Tensor([0])
                        for j in range(len(self.classes)):
torch.LongTensor([self.classes[j]]).reshape([1]).to(self.myDevice)) #predict_label[m,:] in a tensor of size N * K which includes logits related to each class for each data point. cross entropy receives this tensor and the true label and calculates the loss
                            logits = predict_label[m].unsqueeze(0)
                            target = torch.tensor([self.classes[j]], dtype=torch.long).to(self.myDevice)
                            temp_value = myLoss(logits, target)
                            thr_inital = thr_inital.to(self.myDevice) + temp_value.to(self.myDevice)   
                        ave_loss = thr_inital/len(self.classes)
                        all_loss = ori_loss - ave_loss #peer loss value
                        if (all_loss < Initial_threshold): 
                            self.loss_record_mat_ns[m + data_pre_num, epoch] = 1 # this data point is detected as clean
                            self.loss_value_clean= loss_value_clean.to(self.myDevice) + ori_loss.to(self.myDevice) # keeps hc of clean instances
                        self.loss_record_mat_all[m + data_pre_num, epoch] = all_loss.detach()  #all loss is peer loss value
                else:
                    break
            print("Finish training noise detection model")   
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time for noise detection: {execution_time} seconds")
            
            start_time2 = time.time()
            cluster_centroid = {}
            loss_ori_list = list(self.loss_record_mat_ori[:, 0]) #it means the loss of first epoch (contains all instance)
            predict_label_list = list(self.loss_record_mat_pre_label[:, 0])
            for m in range(len(loss_ori_list)):
                label_value = self.y[m]
                if (label_value not in cluster_centroid.keys()): #it is creating cluster centroids and trying to find the centroids with the least loss value
                    cluster_centroid.setdefault(label_value, [self.x[m], loss_ori_list[m]])

                else:
                    if (loss_ori_list[m] < cluster_centroid[label_value][1]):
                        cluster_centroid[label_value] = [self.x[m],  loss_ori_list[m]]
            ns_data_index, ns_label, ns_data = ns_detect(self.loss_record_mat_ns, self.y, self.data_sample, epoch,self.numFeatures)
            
            Xn.extend(ns_data_index)
            cf_correct_data = [] 
            myLoss_cf = torch.nn.CrossEntropyLoss()
            for z in range(len(ns_data)):
                pos_label_list = self.classes
                cf_info = {}
                data_temp = ns_data[z, :]
                test_data = data_temp 
                X_unchange = test_data
                for k in range(len(self.classes)):
                    input_data = cluster_centroid[k][0].unsqueeze(0).to(self.myDevice)
                    tlnn2 = RESNET_MF(input_data)
                    optimizer_cf = torch.optim.AdamW(tlnn2.parameters(), lr = 0.2)
                    Xpred1, cf_data = tlnn2(cf_model)  
                    _, predict_label_cf = Xpred1.max(1)
                    label_pos_list = [self.classes[k]]
                    dis_loss_list = []
                    valid_loss_list = []
                    cf_data_index = []
                    count_num = 0
                    vali_loss = 10
                    predict_label_cf = k
                
                    while  (count_num < 20 and predict_label_cf == k) :
                        Xpred1, cf_data = tlnn2(cf_model)  
                        _, predict_label_cf = Xpred1.max(1)
                        dis_loss = torch.norm(cf_data - torch.tensor(X_unchange, dtype=torch.float), 2)  # Euclidean distance
                        if (predict_label_cf == k):
                            dis_loss_list.append(dis_loss.detach())
                            cf_data_index.append(cf_data.detach())
                        loss_cf = dis_loss 
                        optimizer_cf.zero_grad()
                        loss_cf.backward(retain_graph=True)
                        optimizer_cf.step()
                        count_num = count_num + 1
                    cf_info.setdefault(pos_label_list[k], {"loss": loss_cf.detach(), "dis":dis_loss.detach(), "dis_track":dis_loss_list,  "cf_data":cf_data_index})
                cf_correct_data.append(cf_info) 
            end_time2 = time.time()
            execution_time2 = end_time2 - start_time2
            print(f"Execution time for counterfactual part: {execution_time2} seconds")
            ###now we have counterfactuals of different classes for "all noisy" data. In next step, for each data instance. WHICH COUNTERFACTUAL is more suitable? 
##cf_correct_data:is a list consisting t element (t is the number of detected noisy labels) each element is a dictionary and is associated with one of the detected noisy labeled data instances, and each dictionary contains a dictionary of counterfactuals information

            cf_label, y_train_ns_next, loss_value_noise = cf_search_correct_label(cf_correct_data, ns_data_index,self.myDevice, self.y)
            x_tensor = torch.stack(self.x)  # Convert list of tensors to a single tensor
            y_tensor = torch.tensor(y_train_ns_next)  # Convert list of labels to a tensor

            self.train_dataset= TensorDataset(x_tensor, y_tensor)
            print("The number of changed labels by counterfactual is {}".format(sum(1 for x , y in zip(prev_y_tensor,y_tensor) if x!=y)))
            changed_index=[index for index, (elem1,elem2) in enumerate(zip(prev_y_tensor,y_tensor)) if elem1!=elem2]
           
            #To train with just changed data points
            mismatched_indices = (prev_y_tensor != y_tensor).nonzero().view(-1)

            # Creates a subset of the original dataset with mismatched indices
            changed_dataset = Subset(self.train_dataset, mismatched_indices)
            updated_data_loader = DataLoader(changed_dataset, batch_size=32, shuffle=True)
            
            if prev_y_tensor is not None and torch.allclose(prev_y_tensor.float(), y_tensor.float(), atol=1e-6):
                print("Labels have not changed within tolerance. Stopping the loop.")
                break
            prev_y_tensor = y_tensor.clone() 
            
            #train the model f with the updated dataset
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(cf_model.parameters(), lr=0.001)
            num_epochs = 50  
            for epoch in range(num_epochs):
                for inputs, labels in updated_data_loader:
                    labels = labels.long()
                    optimizer.zero_grad()
                    outputs = cf_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Save the updated model
            torch.save(cf_model.state_dict(), 'FFNeuralNetwork')
            Initial_threshold=update_threshold(cf_model,last_lc_list,y_pre,data_pretrain,self.myDevice,clean_index_pre, noisy_index_pre,ln2_list,new_dataset)
            print("new threshold is", Initial_threshold)

        return cf_label, y_train_ns_next





