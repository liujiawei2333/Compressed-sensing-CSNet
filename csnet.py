import torch
import torch.nn as nn
import numpy as np

def tensor_concat(f,n,batch_size,ngpu):
    a1 = torch.FloatTensor(batch_size//ngpu,1,32,32)
    a1 = f[:,n,:,:]
    for i in range(n+1,n+8):
        a1 = torch.cat((a1,f[:,i,:,:]),-2)
    return a1

def block2image(initail_map,batch_size,ngpu):
    f = initail_map.view(batch_size//ngpu,1024,64)
    f = f.view(batch_size//ngpu,32,32,64)
    f = f.permute(0,3,1,2)
    x1 = tensor_concat(f,0,batch_size//ngpu,ngpu)
    x2 = tensor_concat(f,8,batch_size//ngpu,ngpu)
    x3 = tensor_concat(f,16,batch_size//ngpu,ngpu)
    x4 = tensor_concat(f,24,batch_size//ngpu,ngpu)
    x5 = tensor_concat(f,32,batch_size//ngpu,ngpu)
    x6 = tensor_concat(f,40,batch_size//ngpu,ngpu)
    x7 = tensor_concat(f,48,batch_size//ngpu,ngpu)
    x8 = tensor_concat(f,56,batch_size//ngpu,ngpu)
    x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),-1)
    x = torch.unsqueeze(x,1)
    x = x.permute(0,1,3,2)
    return x

class CSNET(nn.Module):

    def __init__(self,channels,cr):
        super(CSNET,self).__init__()

        self.channels = channels
        self.fcr = 153
        self.base = 1

        self.sample = nn.Conv2d(self.channels,self.fcr,kernel_size=32,padding=0,stride=32,bias=False)
        self.initial = nn.Conv2d(self.fcr,3072,kernel_size=1,padding=0,stride=1,bias=False)
        self.pixelshuffle = nn.PixelShuffle(32)
        self.conv1 = nn.Conv2d(self.channels,self.base,kernel_size=3,padding=1,stride=1,bias=False)
        self.conv2 = nn.Conv2d(self.base,self.base,kernel_size=3,padding=1,stride=1,bias=False)
        self.conv3 = nn.Conv2d(self.base,self.channels,kernel_size=3,padding=1,stride=1,bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,input,batch_size,ngpu):
        output = self.sample(input)
        output = self.initial(output)
        output = self.pixelshuffle(output)
        output = self.relu(self.conv1(output))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv2(output))
        output = self.conv3(output)

        return output