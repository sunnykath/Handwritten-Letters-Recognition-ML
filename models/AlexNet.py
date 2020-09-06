import torch
import torch.nn as nn
import torch.nn.functional as F

#creating neural network class:
class AlexNet(nn.Module):
    #defining attributes of neural networks i.e how many layers, layer types
    def __init__(self):
        super(AlexNet, self).__init__()
        #1 input channel(b/w image),6 output channels, 3x3 filter size
        self.conv1 = nn.Conv2d(1, 6, 3)     
        #6 input channels(feature maps from prev conv2d()call),12 output channels, padded with one layer of zeroes
        #this is done to keep output channels a reasonable resolution/size
        self.conv2 = nn.Conv2d(6, 12, 3, padding = 1)    
        #12 input channels(feature maps from prev conv2d()call),24 output channels, padded with one layer of zeroes
        self.conv3 = nn.Conv2d(12, 24, 3, padding = 1)
        #24 input channels(feature maps from prev conv2d()call),48 output channels, padded with one layer of zeroes
        self.conv4 = nn.Conv2d(24, 48, 3, padding = 1)
        #48 input channels(feature maps from prev conv2d()call),96 output channels, padded with one layer of zeroes
        self.conv5 = nn.Conv2d(48, 96, 3, padding = 1)
        # 96*3*3 input nodes from last conv layer, 764 output nodes
        self.fc1 = nn.Linear(96*3*3, 764)  
        #764 input nodes, 664 output nodes
        self.fc2 = nn.Linear(764, 664)
        #output layer has final class predictions
        #664 input nodes, 26 output nodes(one for each class)
        self.out = nn.Linear(664, 26)
        
    #function that calls layer instances, uses relu and max pool to reduce dimensionality for inputs
    def forward(self, x):
        #Convulational layer one
        #call conv1() on self, passing input tensor as parameter to be convolved into 6*26*26
        x = self.conv1(x)
        #call relu(), passing convovled tensor to have negative values turned to zero
        x = F.relu(x)   
        #pooling reduces dimensionality, reduces number of parameters, and reduces overfitting 
        #takes average value from a 2x2 shape and creates new feature map with reduces dimensionality
        #2x2 avg pool with stride of 2, reduces dimensionality of output from relu to 6*13*13
        x = F.max_pool2d(x, 2, 2)
        
        #Convulational layer two
        #convolved into 12*13*13 size output, padding keeps dimensions the same
        x = self.conv2(x)
        x = F.relu(x)
        #3x3 max pool with stride of 2, reduces dimensionality of output from relu to 12*6*6
        x = F.max_pool2d(x, 3, 2)   
        
        #Convulational layer three
        #convolved into 24*6*6, padding keeps dimensions the same
        x = self.conv3(x)   
        x = F.relu(x)
        
        #Convulational layer four
        #convolved into 48*6*6, padding keeps dimensions the same
        x = self.conv4(x)   
        x = F.relu(x)
        
        #Convulational layer five
        #convolved into 96*6*6, padding keeps dimensions the same
        x = self.conv5(x)   
        x = F.relu(x)
        #2x2 max_pool, reduces dimensionality of output from relu to 96*3*3
        x = F.max_pool2d(x, 2, 2)
        
        #Fully connected layer 1
        #turns 96*3*3 tensor into 1 dimensional tensor with 432 values
        #calls fc1() on self, transitioning from 2d convolution to 1d dense/fully connected 
        x = self.fc1(x.reshape(-1, 96*3*3))  
        x = F.relu(x)
        
        #Fully connected layer 2
        x = self.fc2(x)
        x = F.relu(x)
        
        #Output layer
        x = self.out(x)
        return x
