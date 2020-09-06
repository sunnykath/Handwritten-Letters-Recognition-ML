import torch
import torch.nn as nn
import torch.nn.functional as F


#creating neural network class:
class LeNet(nn.Module):
    #defining attributes of nueral netwroks i.e how many layers, layer types
    
    def __init__(self):
        super(LeNet, self).__init__()
        #nn.conv2d() recognises local patterns, and is able to find edges and lines
        #1 input channel(b/w image),6 output channels, 5x5 filter size 
        self.conv1 = nn.Conv2d(1, 6, 5)
        #6 input channels(feature maps from prev conv2d()call),12 output channels, 5x5 filter size 
        self.conv2 = nn.Conv2d(6, 12, 5)
        #nn.linear() 
        #12*4*4 input nodes(12 feature maps, 4*4 size feature map), 86 output nodes
        self.fc1 = nn.Linear(12 * 4 * 4, 86)
        #96 input nodes, 78 output nodes
        self.fc2 = nn.Linear(86, 76)
        #output layer has final class predictions
        #76 input nodes, 26 output nodes(one for each class)
        self.out = nn.Linear(76, 26)
        
    #function that calls layer instances, uses tanh and max pool to reduce dimensionality for inputs
    def forward(self, x):
        #Convulational layer one
        #call conv1() on self, passing input tensor as parameter to be convolved into 6*24*24
        x = self.conv1(x) 
        #call torch.tanh() , passing convovled tensor to have no nuerons with values >1 or <-1 
        x = torch.tanh(x)  
        #pooling reduces dimensionality, reduces number of parameters, and reduces overfitting 
        #takes average value from a 2x2 shape and creates new feature map with reduces dimensionality
        #2x2 avg pool with stride of 2, reduces dimensionality of output from tanh to 6*12*12 
        x = F.avg_pool2d(x, 2, 2)   
        
        #Convulational layer two
        #convolved into 12*8*8 size output
        x = self.conv2(x)   
        x = torch.tanh(x)
        #reduces dimensionality of output from tanh to 12*4*4
        x = F.avg_pool2d(x, 2, 2)  
        
        #Fully connected layer 1
        #turns 12*4*4 tensor into 1 dimensional tensor with 192 values
        #calls fc1() on self, transitioning from 2d convolution to 1d dense/fully connected 
        x = self.fc1(x.reshape(-1, 12*4*4))  
        x = torch.tanh(x)
        
        #Fully connected layer 2
        #86 input values to 76 output values
        x = self.fc2(x)
        x = torch.tanh(x)
        
        #Output layer
        #76 input to 26 output nodes, i.e one for every class
        x = self.out(x) 
        return x
