import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd
import numpy as np
import os
import shutil

from utils import My_dataset
from model import AttnVGG, user_net

from catalyst.callbacks import SchedulerCallback, CheckpointCallback
from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.contrib.nn.criterion.dice import DiceLoss

class mix_model(nn.Module):
    def __init__(self, user_net, event_net):
        super(mix_model, self).__init__()
        
        self.user_net = user_net
        self.event_net = event_net
        
        
    def forward(self, user, event):
        
        user_feature = self.user_net(user)
        event_feature = self.event_net(event)
        

        
        predict = torch.matmul(user_feature, event_feature.transpose(0, 1))
        
        return torch.sigmoid(predict)


class CustomRunner(dl.Runner):
    def __init__(
        self,
        logdir,
        user_remain,
        user_remain_event,
        epochs,
        ):
        super().__init__()
        self.epochs = epochs
        self._logdir = logdir
        self.user_remain = user_remain
        self.user_remain_event = user_remain_event
        
    
    def get_loggers(self):
        return {
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }
    
    def get_loaders(self, stage: str):
        train_dataset = My_dataset(self.user_remain[:800], self.user_remain_event)
        test_dataset = My_dataset(self.user_remain[800:1000], self.user_remain_event)
        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
            
            valid_sampler = DistributedSampler(
                test_dataset,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None

        
        train_loader = DataLoader(
            dataset=train_dataset, 
            sampler=train_sampler,
            pin_memory=True,
            num_workers=1,
            )
        
        test_loader = DataLoader(
            dataset=test_dataset, 
            sampler=valid_sampler,
            pin_memory=True,
            num_workers=1,
            )   
        
        loaders = {
            "train": train_loader,
            "valid": test_loader,
        }

        return loaders
    
    def get_scheduler(self, stage: str, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        
    def handle_batch(self, batch):
        user_tensor, event_matrix_train, event_matrix_test,  train_target, test_target = batch
        predict1 = model(user_tensor, event_matrix_train)
        predict2 = model(user_tensor, event_matrix_test)
        
        loss1 = F.mse_loss(predict1, train_target)
        loss2 = F.mse_loss(predict2, test_target)
        loss = (loss1 + loss2)/2
        
        self.batch_metrics.update(
            {"loss": loss, "loss1":loss1, "loss2":loss2}
        )
        

if __name__=="__main__":
    ## dataframe of event and users from .csv file
    event = pd.read_csv('/data/users2/yxiao11/class_pj/Computer_network/event.csv')
    users = pd.read_csv('/data/users2/yxiao11/class_pj/Computer_network/attendee_event.csv')

    ## return is numpy array with single dimension value
    user_index = users['Unnamed: 0'].values
    user_event = users['Event'].values

    ## get users attended at least 2 meetings
    delete_index = []
    for user in user_index:
        events_per_user = user_event[user][1:-1].split(', ')
        if len(events_per_user) < 2:
            delete_index.append(user)
    user_remain = np.delete(user_index, delete_index)
    user_remain = np.arange(len(user_remain))
    user_remain_event = np.delete(user_event, delete_index) # the relative events each user has attended
    
    

    
    # user_net = AttnVGG(in_channels=4, out_channels=300, final_channels=100)
    # event_net = AttnVGG(in_channels=4, out_channels=300, final_channels=100)
    
    # model = mix_model(
    #     AttnVGG(in_channels=4, out_channels=300, final_channels=100), 
    #     AttnVGG(in_channels=4, out_channels=300, final_channels=100),
    #     )
    
    model = mix_model(
        user_net(1, 300),
        user_net(1, 300)
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 
    
    if os.path.exists('./log'):
        shutil.rmtree('./log')
    
    logdir = './log'
    
    
    runner = CustomRunner(
        logdir=logdir,
        user_remain=user_remain,
        user_remain_event=user_remain_event,
        epochs=30
    )
    
    runner.train(
        model=model, 
        criterion=None, 
        optimizer=optimizer,  
        loaders=None, 
        callbacks=[
            CheckpointCallback(logdir=logdir),
            dl.OptimizerCallback(
                metric_key="loss",
                ),
            dl.SchedulerCallback(),
            ],
        logdir=logdir, 
        num_epochs=30, 
        verbose=True,
        ddp=True,
        amp=False,
        )
    
    
    # num_epoch = 30
    
    # for epoch in range(num_epoch):
    #     for step, batch in enumerate(loader):
    #         user_tensor, event_matrix_train, event_matrix_test,  train_target, test_target = batch
    #         user_tensor = user_tensor.to(device)
    #         event_matrix_train = event_matrix_train.to(device)
    #         event_matrix_test = event_matrix_test.to(device)
    #         train_target = train_target.to(device)
    #         test_target = test_target.to(device)
            
            
    #         predict1 = model(user_tensor, event_matrix_train)
    #         predict2 = model(user_tensor, event_matrix_test)
            
    #         loss1 = criterion(predict1, train_target)
    #         loss2 = criterion(predict2, test_target)
    #         loss = (loss1 + loss2)/2
            
    #         print(loss)
            
    #         loss.backward()
    #         optimizer.step()
        
    
    
    
    # opt = model(a[0], a[1])
    
    # print(opt)