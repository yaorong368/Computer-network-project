import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import convlayer
from attention import ProjectorBlock, SpatialAttn, TemporalAttn

import math

class user_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(user_net, self).__init__()

        out_channels_l = int(out_channels//3)
        self.conv1 = convlayer(in_channels, out_channels_l, 1)
        self.act1 = nn.Sigmoid()
        self.conv2 = convlayer(in_channels, out_channels_l, 3)
        self.act2 = nn.Sigmoid()
        self.conv3 = convlayer(in_channels, out_channels_l, 5)
        self.act3 = nn.Sigmoid()
        self.linear = nn.Linear(out_channels, 20)
        self.act4 = nn.Sigmoid()
            
            
    def forward(self, x):
        x1 = self.act1(self.conv1(x))
        x2 = self.act2(self.conv2(x))
        x3 = self.act3(self.conv3(x))
        opt = torch.cat([x1,x2,x3], dim=1) #(1,90,*,1)
        opt, _ = torch.max(opt, dim=2)
        opt = self.act4(self.linear(opt.squeeze(-1)))
        return opt
    
    
    
class AttnVGG(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        final_channels,
        normalize_attn=True, 
        init_weights=True,
        ):
        super(AttnVGG, self).__init__()
        # conv blocks
        self.conv1 = self._make_layer(1, 64, 2)
        self.conv2 = self._make_layer(64, 128, 2)
        self.conv3 = self._make_layer(128, 256, 3)
        self.conv4 = self._make_layer(128, 512, 3)
        self.conv5 = self._make_layer(128, 512, 3)
        self.conv6 = self._make_layer(128, 512, 2)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True)
        # attention blocks

        self.projector = ProjectorBlock(256, 512) ##1*1 kernel, no shape change
        self.attn1 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
        self.attn2 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
        self.attn3 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
        # final classification layer

            # self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
        self.final_layer = nn.Conv2d(512*3, 1, kernel_size=3, padding=1)
        
        
        out_channels_l = int(out_channels//3)
        self.conv7 = convlayer(in_channels, out_channels_l, 1)
        self.conv8 = convlayer(in_channels, out_channels_l, 3)
        self.conv9 = convlayer(in_channels, out_channels_l, 5)
        self.projector2 = nn.Linear(out_channels, final_channels)
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        l1 = self.conv3(x)
        # x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)
        l2 = self.conv4(x)
        # x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)
        l3 = self.conv5(x)
        # x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)
        x = self.conv6(x)
        g = self.dense(x) # batch_sizex512x1x1
        # attention

        c1, g1 = self.attn1(self.projector(l1), g)
        c2, g2 = self.attn2(l2, g)
        c3, g3 = self.attn3(l3, g)
        g = torch.cat((g1,g2,g3), dim=1) # batch_sizex3C
        # classification layer
        x = F.relu(self.final_layer(g)) # batch_sizexnum_classes
        
        # return [x, c1, c2, c3]
        x = torch.cat((x,c1,c2,c3), dim=1)
        
        x1 = self.conv7(x)
        x2 = self.conv8(x)
        x3 = self.conv9(x)
        opt = torch.cat([x1,x2,x3], dim=1)
        opt, _ = torch.max(opt, dim=2)
        opt = self.projector2(opt.squeeze(-1))
        return F.relu(opt)
        
        # return torch.cat((x,c1,c2,c3), dim=1)

    def _make_layer(self, in_features, out_features, blocks, pool=False):
        layers = []
        for i in range(blocks):
            conv2d = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1, bias=False)
            layers += [conv2d, nn.BatchNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
            if pool:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    
    
    

