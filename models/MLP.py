import torch
import torch.nn as nn
import torch.nn.functional as F


# Create the Neural Network Class
class MultiP(nn.Module):
    # defining attributes of neural networks i.e how many layers, layer types
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(MultiP, self).__init__()
        self.h1 = nn.Linear(input_size, hidden1_size)  # create the first hidden layer
        self.h2 = nn.Linear(hidden1_size, hidden2_size)
        self.out = nn.Linear(hidden2_size, num_classes)
    # function that calls layer instances, uses relu and max pool to reduce dimensionality fo inputs
    # no softmax because training loop will use softmax

    def forward(self, x):
        x = self.h1(x)              # First Hidden Layer
        x = F.relu(x)               # Call ReLU
        x = self.h2(x)              # Second Hidden Layer
        x = F.relu(x)
        x = self.out(x)             # Last/Output Layer
        return x