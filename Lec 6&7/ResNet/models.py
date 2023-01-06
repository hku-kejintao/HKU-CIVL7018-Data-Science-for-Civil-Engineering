from torchvision import models
import torch.nn as nn


def model_A(num_classes):
    # pretrained = True means we use the pretrained parameters of ResNet18
    model_resnet = models.resnet18(pretrained=True)
    num_features = model_resnet.fc.in_features # The input channels of the full connection layer
    model_resnet.fc = nn.Linear(num_features, num_classes) # We modify the number of classes
    # We only train the full connection layer (finetune)
    for param in model_resnet.parameters():
        param.requires_grad = False
    for param in model_resnet.fc.parameters():
        param.requires_grad = True
    return model_resnet


