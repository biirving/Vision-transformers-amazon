

from transformers import BeitFeatureExtractor, BeitForImageClassification
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset
from einops import rearrange, reduce
import torchvision.transforms as transforms
import h5py
from torch import nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
from torch.optim import AdamW
from datasets import load_dataset, load_metric
import numpy as np
import tarfile
import imageNet
import torchvision.datasets
from datasets import load_dataset
from PIL import Image
from einops import repeat, rearrange
from torchvision.io import read_image
# import required module
from pathlib import Path

# import the modules
import os
from os import listdir

"""
This file is for measuring the accuracy of the models on GPU. Select which model you wish to measure, and run it on the Imagenet dataset.
"""
model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
#model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# data preparation


# Set filepath based on where you store the text file for the Imagenet dataset, which will contain 
text_file = open("your-file-path", "r")


labels = text_file.readlines()
labels_final = []
for l in labels:
    temp = l.split(' ')[1:2]
    add = temp[0].split('\n')[0]
    labels_final.append(int(add))

text_file.close()



my_transforms = transforms.Compose([
transforms.ToPILImage(),                                                                                                                                                                                                   
transforms.Resize((224, 224)),                                                                                                  
transforms.ToTensor()                                                                                                           
])


huh = []
num_images = 10000

image_index = 1
greyscale = []
print("Preparing images")

def getDigits(image_index):
    length = len(str(image_index))
    zeros = (8 - length) * '0'
    return zeros + str(image_index)

# Validation Input
for x in range(num_images):
    # Set the file path here to where you have stored the imagenet validation set
    img =  read_image('your file path to the imagenet validation set'+'/ILSVRC2012_val_' + getDigits(image_index) + '.JPEG')
    toAdd = my_transforms(img)
    toAdd = toAdd[None, :, :, :]
    if(toAdd.shape != torch.Size([1, 3, 224, 224])):
        greyscale.append(image_index)
        toAdd = repeat(toAdd, 'b c h w -> b (3 c) h w')
    toAdd = toAdd.to(device)
    huh.append(toAdd)
    image_index+=1


model.eval()

tot = 0
correct = 0
index = 0
print("Evaluation begins")
for m in range(len(huh)):
    woah = torch.max(model(huh[m]).logits, dim = -1).indices
    print(index)
    if(woah[0].item() == labels_final[index]):
        correct += 1
    tot += 1
    index += 1

print(f"eval : {correct/tot}")
