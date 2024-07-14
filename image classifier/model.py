#author:Hieu Tran
import torch
from torch import nn
from torchvision import models

def build_model(arch='vgg16', hidden_units=512, output_size=102):
    if arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported architecture")
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    return model

def save_checkpoint(model, save_dir, arch, hidden_units, class_to_idx, optimizer, epochs):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'], len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
