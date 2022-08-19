

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
from pathlib import Path
import torch.neuron

# import the modules
import os
from os import listdir


"""
The purpose of this file is to test the accuracy of the compiled vision transformer models on Inferentia.
In the original formulation, the ImageNet validation set was used, the labels of which are included in the text file within this folder.
Other datasets may be used, but there will have to be some alterations made to this file. 
"""

# Change this line to adjust for the model that you are using. 
model = torch.jit.load('your-path-to-compiled-models')


# Labels
text_file = open("Vision-transformers-amazon/Inferentia/val_map.txt", "r")
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


all_image_inputs = []
num_images = 10000

image_index = 1
greyscale = []
print("Preparing images")

def getDigits(image_index):
    """
    The purpose of this function is to read from the ImageNet validation set labels. Ignore if a different dataset is being used. 
    """
    length = len(str(image_index))
    zeros = (8 - length) * '0'
    return zeros + str(image_index)

# Validation Input
for x in range(num_images):
    """
    You will have to download the imagenet dataset to your machine or instance, and place in preferrably in this folder to make it accessible.
    If another dataset is to be used, you may change this line accordingly.
    """
    img =  read_image('Your-path-to-imagenet-dataset/ILSVRC2012_val_' + getDigits(image_index) + '.JPEG')
    print(image_index)
    toAdd = my_transforms(img)
    toAdd = toAdd[None, :, :, :]
    if(toAdd.shape != torch.Size([1, 3, 224, 224])):
        greyscale.append(image_index)
        toAdd = repeat(toAdd, 'b c h w -> b (3 c) h w')
    all_image_inputs.append(toAdd)
    image_index+=1

model.eval()

tot = 0
correct = 0
index = 0
print("Evaluation begins")
for m in range(len(huh)):

    max_index = torch.max(model(huh[m])['logits'], dim = -1).indices
    print(index)
    if(max_index[0].item() == labels_final[index]):
        correct += 1
    tot += 1
    index += 1

print(f"eval : {correct/tot}")
                                          


