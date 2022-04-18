import torch
import torch.nn as nn
import torch.nn.functional as F





class convlayer(nn.Module):
    '''
    this layer is to get the kernel of 3, 4, 5 by width kernel
    '''
    def __init__(self, in_channels, out_channels, num_words, embedding_dim = 3):
        super(convlayer, self).__init__()
        
        self.num_words = num_words
        self.weight = torch.zeros(out_channels, in_channels, num_words, 100, requires_grad=True)
        self.bias = torch.rand(out_channels, requires_grad=True)*0.1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1)
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.weight)
            self.conv.bias = nn.Parameter(self.bias)
            
    def forward(self, x):
        if self.num_words == 3:
            x = F.pad(x,(0,0,1,1))
        elif self.num_words == 5:
            x = F.pad(x, (0,0,2,2))
        
        x = self.conv(x)
        return F.relu(x)
    
    