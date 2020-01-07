## TODO: define the convolutional neural network architecture
## Research Paper from NaimishNet :: https://arxiv.org/pdf/1710.00977.pdf
## Video Tutorial explaining cnn and filters:: https://www.youtube.com/watch?v=umGJ30-15_A
## https://www.youtube.com/watch?v=YRhxdVk_sIs
## https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
       # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv1 = nn.Conv2d(1,32,5) # (32,220,220) output tensor # (W-F)/S + 1 = (224-5)/1 + 1 = 220
        self.pool1 = nn.MaxPool2d(2,2) #(32,110,110)
        
        # second convolutional layer
        self.conv2 = nn.Conv2d(32,64,3) # (64,108,108) output tensor # (W-F)/S + 1 = (110-3)/1 + 1 = 108
        # second Max-pooling layer
        self.pool2 = nn.MaxPool2d(2,2) # (64,54,54) output tensor
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64,128,3)# (128,52,52) output tensor # (W-F)/S + 1 = (54-3)/1 + 1 = 52
        self.pool3 = nn.MaxPool2d(2,2) # (128,26,26) output tensor
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(128,256,3)# (256,24,24) output tensor # (W-F)/S + 1 = (26-3)/1 + 1 = 24
        self.pool4 = nn.MaxPool2d(2,2) # (256,12,12) output tensor
        
        
        # Fully connected layer
        self.fc1 = nn.Linear(256*12*12, 1024)   
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 136)
           
        # Dropouts
        self.drop1 = nn.Dropout(p = 0.25)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        #x = self.drop1(x)
        #print(x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        #x = self.drop2(x)
        #print(x.shape)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        #x = self.drop3(x)
        #print(x.shape)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        #x = self.drop4(x)
      
        # Flatten before passing to the fully-connected layers.
        #print(x.size(0))
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)
        
       
        # First - Dense + Activation + Dropout
        x = self.drop1(F.relu(self.fc1(x)))
        #print("First dense size: ", x.shape)

        # Second - Dense + Activation + Dropout
        x = self.drop1(F.relu(self.fc2(x)))
        #print("Second dense size: ", x.shape)

        # Final Dense Layer
        x = self.fc3(x)
        #print("Final dense size: ", x.shape)

        # a modified x, having gone through all the layers of your model, should be returned
        return x