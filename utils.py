import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import random
import os


class My_dataset(Dataset):
    def __init__(
        self, 
        user_remain, # here we use user_remain 
        user_remain_event, # here we use user_remain_event
        event_matrix_dir = '/data/users2/yxiao11/class_pj/Computer_network/data/event_matrix/', # dir '/data/users2/yxiao11/class_pj/Computer_network/data/event_matrix'
        ):
        
        self.user_remain = user_remain
        self.user_remain_event = user_remain_event
        self.event_matrix_dir = event_matrix_dir
        
        pass
    
    def __len__(self):
        return len(self.user_remain)



    def __getitem__(self, index):
        input_list=[]
        target_list=[]
        target_name_list=[]
        
        user_id = self.user_remain[index]
        event_per_user = self.user_remain_event[user_id][1:-1].split(', ')
        
        input_event_list = event_per_user[:len(event_per_user)//2]
        target_event_list = event_per_user[len(event_per_user)//2:]
        
        
        for event in input_event_list:
            file_name = event[1:-1] + '.npy'
            path_i = self.event_matrix_dir + file_name # get the relitive event matrix
            input_list.append(np.load(path_i))
            # target_list.append(np.load(path_i))
    
        for event in target_event_list:
            file_name = event[1:-1] + '.npy'
            target_name_list.append(file_name)
            path_i = self.event_matrix_dir + file_name # get the relitive event matrix
            # target_list.append(np.load(path_i))
            
        # input_array = np.concatenate(input_list)
        # target_array = np.concatenate(target_list)
        
        user_tensor = torch.from_numpy(np.expand_dims(np.concatenate(input_list), 0)).to(torch.float32)
        # target_tensor =  torch.from_numpy(np.expand_dims(np.concatenate(target_list), 0)).to(torch.float32)
        
        test = random.choice(os.listdir(self.event_matrix_dir))
        train = random.choice(target_name_list)
        
        if test in train:
            test_target = torch.tensor([1.0])
        else:
            test_target = torch.tensor([0.0])
        train_target = torch.tensor([1.0])
        event_matrix_train = np.expand_dims(np.load(self.event_matrix_dir+train), 0)
        event_matrix_test = np.expand_dims(np.load(self.event_matrix_dir+test), 0)
        
        event_matrix_train = torch.from_numpy(event_matrix_train).to(torch.float32)
        event_matrix_test = torch.from_numpy(event_matrix_test).to(torch.float32)
        
        return user_tensor, event_matrix_train, event_matrix_test,  train_target, test_target
    
    