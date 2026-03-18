import torch
import torch.nn as nn
from torchvision import models

class CheXpertResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feature_maps = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(feature_maps)
        x = torch.flatten(x, 1)
        logits = self.backbone.fc(x)
        
        return logits, feature_maps
