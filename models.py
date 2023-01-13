from torch import nn

### BASIC MODEL ###
# model definition
class BasicModel(nn.Module):
    # define model elements
    def __init__(self):
        super(BasicModel, self).__init__()
        # define layer
        # data input dimensions, no. output features, weight dimension => 64 x 224 x 224 (batch size x 64 x 2D)
        self.layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # hỗ trơ tính gradient => 64 x 224 x 224 (batch x 64 x 2D)
        self.activation = nn.ReLU()
        # reduce size of features => 64 x 112 x 112 (batch x 64 x 2D)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # transform from 64x112x112 (batchsize x 1D)
        self.flatten = nn.Flatten()
        # 2nd layer fully connected (fc)
        self.fc = nn.Linear(in_features= 64*112*112, out_features=2)
        # đưa về tỉ lệ (100%)
        self.softmax = nn.Softmax()
 
    # forward propagate input
    def forward(self, X):
        # define data pipeline
        X = self.layer(X)
        X = self.activation(X)
        X = self.maxpool(X)
        X = self.flatten(X)
        X = self.fc(X)
        X = self.softmax(X)
        return X

class ConvBlock(nn.Module):
    # define model elements
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.ReLU()
    # forward propagate input
    def forward(self, X):
        # define data pipeline
        X = self.layer(X)
        X = self.activation(X)
        return X

class VGGModel(nn.Module):
    # define model elements
    def __init__(self):
        super(VGGModel, self).__init__()
        # define block
        self.block1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features= 512*14*14, out_features=4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features= 4096, out_features=1000)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features= 1000, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # define data pipeline
        X = self.block1(X)
        X = self.maxpool1(X)
        X = self.block2(X)
        X = self.maxpool2(X)
        X = self.block3(X)
        X = self.maxpool3(X)
        X = self.block4(X)
        X = self.maxpool4(X)
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.relu1(X)
        X = self.fc2(X)
        X = self.relu2(X)
        X = self.fc3(X)
        X = self.softmax(X)
        return X

from torchvision.models import resnet18, ResNet18_Weights
class Resnet18Model(nn.Module):
    def __init__(self):
        super(Resnet18Model, self).__init__()
        self.model = nn.Sequential(
            resnet18(weights=ResNet18_Weights.DEFAULT),
            nn.Linear(1000,2),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        X = self.model(X)
        return X