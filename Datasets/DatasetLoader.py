import os
import json
from typing import Any
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

vgRoot = 'Datasets/VisualGenome/'

vgAttr = vgRoot+'Annotation/attributes.json'
vgSG = vgRoot+'Annotation/scene_graphs.json'

vgExAttr = 'ExtractAttribute'
vgExRel = 'ExtractRelation'
vgImg = 'ExtractImage'
vgGT = 'ExtractGT'

resize = (224,224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class MyTransform():
    def __init__(self, resize, mean, std):
        self.imageTrans = {
            "train": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        }
    
    def __call__(self, image, mode = "train"):
        return self.imageTrans[mode](image)


class DatasetLoader(Dataset):
    def __init__(self, imageList, transform = None, mode = 'train'):
        self.imageList = imageList
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, index):
        imagePath = self.imageList[index]
        print(imagePath)
        image = Image.open(imagePath)
        imageTransform = self.transform(image, self.mode)

        annotation = os.path.join(imagePath.replace(vgImg, vgGT).replace('.jpg', '.json'))
        annotation = open(annotation)
        annotation = json.load(annotation)
        sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel = [],[],[], [],[],[], [] 

        for item in annotation:
            sub.append(item['id_sub'])
            subBbox.append(item['sub_bbox'])
            attributeSub.append(item['att_sub'])

            obj.append(item['id_obj'])
            objBbox.append(item['obj_bbox'])
            attributeObj.append(item['att_obj'])

            rel.append(item['rel'])

        return imageTransform, sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # chuyển từ tensor sang numpy
    plt.imshow(img)
    plt.axis('off')  # không hiển thị trục

def GetAllImage(imageDir):
    imageList = []
    for item in os.listdir(imageDir):
        #imageList.append(os.path.join(vgRoot + vgImg, item))
        imageList.append(vgRoot + vgImg + '/' + item)
    return imageList[:]

trainList = GetAllImage(vgRoot+vgImg)

trainDataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode='train')

print(trainDataset.__len__())
index = 0
imageTransform, sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel = trainDataset.__getitem__(index)
print(imageTransform.shape)
print('ID Subjects: ', sub)
print('BBox Subjects: ', subBbox)
print('Attribute Subjects: ', attributeSub)
print('Relation: ', rel)

imshow(imageTransform)

plt.show()