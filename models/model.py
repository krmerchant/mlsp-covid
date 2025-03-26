import torch
import torchvision.models as models
import torch.nn as nn

class CustomConvNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(CustomConvNet, self).__init__()
        
        # Load a pretrained ResNet model (you can choose ResNet18, ResNet34, ResNet50, ResNet101, etc.)
        self.resnet = models.resnet50(pretrained=pretrained)  # You can change this to other ResNet variants
               # Modify the final fully connected layer to match the number of classes in your task
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)  # num_classes is the output dimension (10 by default)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        resnet_features = self.resnet(x)
        print(f'{resnet_features.shape=}') 
        classes = self.softmax(resnet_features)
        return classes
