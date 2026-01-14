
from torchinfo import summary
import torch.nn as nn


class ModelFullyconnected(nn.Module):

    def __init__(self):
        super(ModelFullyconnected, self).__init__()  # call the parent constructor

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define the layers of the model
        self.fc = nn.Linear(ninputs, noutputs)

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def forward(self, x):

        # print('Forward method called ...')
        # print('Input x.shape = ' + str(x.shape))

        # flatten the input to a vector of 1x28x28
        x = x.view(x.size(0), -1)
        # print('Input x.shape = ' + str(x.shape))

        # Now we can pass through the fully connected layer
        y = self.fc(x)
        # print('Output y.shape = ' + str(y.shape))

        return y

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelConvNet(nn.Module):

    def __init__(self):

        super(ModelConvNet, self).__init__()  # call the parent constructor

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # this will output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 32x14x14

        # Define second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # this will output 64x14x14

        # Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 64x7x7

        # Define the first fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # this will output 128

        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # this will output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        print('Forward method called ...')

        print('Input x.shape = ' + str(x.shape))

        x = self.conv1(x)
        print('After conv1 x.shape = ' + str(x.shape))

        x = self.pool1(x)
        print('After pool1 x.shape = ' + str(x.shape))

        x = self.conv2(x)
        print('After conv2 x.shape = ' + str(x.shape))

        x = self.pool2(x)
        print('After pool2 x.shape = ' + str(x.shape))

        # Transform to latent vector
        x = x.view(-1, 64*7*7)
        print('After flattening x.shape = ' + str(x.shape))

        x = self.fc1(x)
        print('After fc1 x.shape = ' + str(x.shape))

        y = self.fc2(x)
        print('Output y.shape = ' + str(y.shape))

        return y


class ModelConvNet3(nn.Module):
    """This is a more complex ConvNet model with 3 conv layers."""

    def __init__(self):

        super(ModelConvNet3, self).__init__()  # call the parent constructor

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # this will output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 32x14x14

        # Define second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # this will output 64x14x14

        # Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 64x7x7

        # Define second conv layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        # this will output ?

        # Define the second pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output ?

        # Define the first fully connected layer
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        # this will output 128

        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # this will output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        # print('Forward method called ...')

        # print('Input x.shape = ' + str(x.shape))

        x = self.conv1(x)
        # print('After conv1 x.shape = ' + str(x.shape))

        x = self.pool1(x)
        # print('After pool1 x.shape = ' + str(x.shape))

        x = self.conv2(x)
        # print('After conv2 x.shape = ' + str(x.shape))

        x = self.pool2(x)
        # print('After pool2 x.shape = ' + str(x.shape))

        x = self.conv3(x)
        # print('After conv3 x.shape = ' + str(x.shape))

        x = self.pool3(x)
        # print('After pool3 x.shape = ' + str(x.shape))

        # Transform to latent vector
        x = x.view(-1, 128*2*2)
        # print('After flattening x.shape = ' + str(x.shape))

        x = self.fc1(x)
        # print('After fc1 x.shape = ' + str(x.shape))

        y = self.fc2(x)
        # print('Output y.shape = ' + str(y.shape))

        return y


class ModelBetterCNN(nn.Module):
    def __init__(self):
        super(ModelBetterCNN, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)  # Dropout for regularization
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third conv block (additional layer)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        
        print('ModelBetterCNN initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))
    
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(-1, 128 * 7 * 7)
        
        # FC layers
        x = self.fc1(x)
        x = self.dropout4(x)
        y = self.fc2(x)
        
        return y
    
    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)