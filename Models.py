from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# First Model with Batch Normalization and relu activation
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second block
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Third block
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Fourth block
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Fifth block
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        
        # Sixth block
        self.conv6 = nn.Conv2d(16, 16, 3)
        
        # Final 1x1 conv
        self.conv7 = nn.Conv2d(16, 10, 1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        
        # Second block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        
        # Third block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        
        # Fourth block
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        
        # Fifth block
        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        
        # Sixth block
        x = self.conv6(x)
        
        # Final 1x1 conv
        x = self.conv7(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)

def get_model1():
    return Net1()

# ======================================================================================================================
# Common function to save the model

def save_model(model, accuracy):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_model_{accuracy:.2f}_{timestamp}.pth"
    torch.save(model.state_dict(), f"models/{filename}")
    return filename

# ======================================================================================================================

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # input: 28x28x1, output: 26x26x8
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # output: 24x24x16
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.15)
        
        # First MaxPool + 1x1 Conv
        self.pool1 = nn.MaxPool2d(2, 2)  # output: 12x12x16
    
        # Third Block
        self.conv3 = nn.Conv2d(16, 8, kernel_size=1)  # output: 12x12x8
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3)  # output: 10x10x16
        self.bn4 = nn.BatchNorm2d(16)
        
        # Fourth Block
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)  # output: 8x8x16
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.15)
        
        # Second MaxPool + 1x1 Conv
        self.pool2 = nn.MaxPool2d(2, 2)  # output: 4x4x16
        self.conv6 = nn.Conv2d(16, 8, kernel_size=1)  # output: 4x4x8
        self.bn6 = nn.BatchNorm2d(8)
        
        # Fifth Block
        self.conv7 = nn.Conv2d(8, 16, kernel_size=3)  # output: 2x2x16
        self.bn7 = nn.BatchNorm2d(16)
        
        # Final Conv
        self.conv8 = nn.Conv2d(16, 10, kernel_size=2)  # output: 1x1x10
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # First Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # First MaxPool + 1x1 Conv
        x = self.pool1(x)
        x = self.conv3(x)
        
        # Third Block
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Fourth Block
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Second MaxPool + 1x1 Conv
        x = self.pool2(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        # Fifth Block
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        # Final Conv
        x = self.conv8(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)

def get_model2():
    return Net2()

# ======================================================================================================================

class Net3(nn.Module):
    def __init__(self):

        super(Net3, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)        # Input: 28x28x1 | Output: 26x26x16 | RF: 3x3
        self.norm1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.15)

        self.conv2 = nn.Conv2d(16, 16, 3)       # Input: 26x26x16 | Output: 24x24x16 | RF: 5x5
        self.norm2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.15)

        self.conv3 = nn.Conv2d(16, 16, 3)       # Input: 24x24x16 | Output: 22x22x32 | RF: 7x7
        self.norm3 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2, 2)          # Input: 22x22x32 | Output: 11x11x32 | RF: 14x14

        self.conv4 = nn.Conv2d(16, 16, 3)       # Input: 9x9x16 | Output: 7x7x16 | RF: 18x18
        self.norm4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.15)

        self.conv5 = nn.Conv2d(16, 16, 3)       # Input: 7x7x16 | Output: 5x5x16 | RF: 20x20
        self.norm5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout(0.15)

        self.conv6 = nn.Conv2d(16, 10, 3)       # Input: 5x5x16 | Output: 3x3x10 | RF: 22x22

        self.gap = nn.AdaptiveAvgPool2d(1)      # Input: 3x3x10 | Output: 1x1x10
    
    def forward(self, x):
        """ This function defines the network structure """
        net = self.dropout1(F.relu(self.norm1(self.conv1(x))))
        net = self.dropout2(F.relu(self.norm2(self.conv2(net))))
        net = self.pool(F.relu(self.norm3(self.conv3(net))))
    
        net = self.dropout4(F.relu(self.norm4(self.conv4(net))))
        net = self.dropout5(F.relu(self.norm5(self.conv5(net))))
        net = F.relu(self.conv6(net))
        net = self.gap(net)
    
        net = net.view(-1, 10)
        return F.log_softmax(net)
    
def get_model3():
    return Net3()

# ======================================================================================================================

class Net4(nn.Module):
    def __init__(self):

        super(Net4, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)      # Input: 28x28x1  | Output: 26x26x16 | RF: 3x3
        self.norm1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(16, 16, 3)     # Input: 26x26x16 | Output: 24x24x16 | RF: 5x5
        self.norm2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)

        self.pool = nn.MaxPool2d(2, 2)        # Input: 24x24x16 | Output: 12x12x16 | RF: 10x10

        self.conv4 = nn.Conv2d(16, 12, 3)     # Input: 12x12x16 | Output: 10x10x16 | RF: 14x14
        self.norm4 = nn.BatchNorm2d(12)
        self.dropout4 = nn.Dropout(0.1)

        self.conv5 = nn.Conv2d(12, 12, 3)     # Input: 10x10x16 | Output: 8x8x16   | RF: 18x18
        self.norm5 = nn.BatchNorm2d(12)
        # self.dropout5 = nn.Dropout(0.1)

        self.conv6 = nn.Conv2d(12, 10, 3)     # Input: 8x8x16   | Output: 6x6x10   | RF: 22x22
        self.gap = nn.AdaptiveAvgPool2d(1)    # Input: 6x6x10   | Output: 1x1x10   | RF: 22x22
    
    def forward(self, x):
        """ This function defines the network structure """
        x = self.dropout1(F.relu(self.norm1(self.conv1(x))))
        x = self.dropout2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(x)  # Directly applying pool after conv2
    
        x = self.dropout4(F.relu(self.norm4(self.conv4(x))))
        # x = self.dropout5(F.relu(self.norm5(self.conv5(x))))
        x = F.relu(self.norm5(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.gap(x)
    
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
    
def get_model4():
    return Net4()
 
# ======================================================================================================================

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 12, 3, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(12, 12, 3, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 20, 3, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Sequential(nn.AvgPool2d(kernel_size=6)),
            nn.Conv2d(20, 10, kernel_size=(1, 1), bias=False)
        )

        self.fc = nn.Sequential(
            nn.Linear(96, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x
    
def get_model5():
    return Net5()

# ======================================================================================================================

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        
        # First convolutional block
        self.conv1_1 = nn.Conv2d(1, 16, 3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.relu1_1 = nn.ReLU()
        self.dropout1_1 = nn.Dropout(0.05)

        self.conv1_2 = nn.Conv2d(16, 16, 3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.relu1_2 = nn.ReLU()
        self.dropout1_2 = nn.Dropout(0.05)

        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional block
        self.conv2_1 = nn.Conv2d(16, 12, 3, bias=False)
        self.bn2_1 = nn.BatchNorm2d(12)
        self.relu2_1 = nn.ReLU()
        self.dropout2_1 = nn.Dropout(0.05)

        self.conv2_2 = nn.Conv2d(12, 12, 3, bias=False)
        self.bn2_2 = nn.BatchNorm2d(12)
        self.relu2_2 = nn.ReLU()

        # Third convolutional block
        self.conv3 = nn.Conv2d(12, 20, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(20)
        self.relu3 = nn.ReLU()

        # Final layers
        self.avgpool = nn.AvgPool2d(kernel_size=6)
        self.conv4 = nn.Conv2d(20, 10, kernel_size=(1, 1), bias=False)
        self.fc = nn.Linear(10, 10)             # Adjusted input size to match output of conv4

    def forward(self, x):
        # First convolutional block
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.dropout1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.dropout1_2(x)

        x = self.pool1(x)

        # Second convolutional block
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.dropout2_1(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.avgpool(x)
        x = self.conv4(x)

        x = x.view(-1, 10)  # Flatten to match the input size of the final layer
        x = F.log_softmax(x, dim=1)
        return x

def get_model6():
    return Net6()    
