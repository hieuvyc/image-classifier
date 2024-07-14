#author:Hieu Tran
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import json

def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
    
    return dataloaders, image_datasets['train'].class_to_idx

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array '''
    im = Image.open(image_path)
    if im.size[0] > im.size[1]:
        im.thumbnail((256, 256*im.size[0]//im.size[1]))
    else:
        im.thumbnail((256*im.size[1]//im.size[0], 256))
    left = (im.width-224)/2
    top = (im.height-224)/2
    right = (im.width+224)/2
    bottom = (im.height+224)/2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def load_category_names(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
