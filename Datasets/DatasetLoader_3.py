import os
import json
from typing import Any
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import Util as ults

vgRoot = 'Datasets/VisualGenome/'

vgImageTrain = 'Train/image'
vgAnnoTrain = 'Train/anno'

vgImageVal = 'Val/image'
vgAnnoVal = 'Val/anno'

resize = (224,224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch = 2

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
    def __init__(self, imageList, annoList, transform = None, mode = 'train'):
        self.imageList = imageList
        self.annoList = annoList
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, index):
        imagePath = self.imageList[index]
        #print(imagePath)
        

        annotation = os.path.join(imagePath.replace(vgImg, vgGT).replace('.jpg', '.json'))
        annotation = open(annotation)
        annotation = json.load(annotation)

        image = Image.open(imagePath)
        originalSize = image.size
        imageTransform = self.transform(image, self.mode)
        
        sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel = [],[],[], [],[],[], [] 

        for item in annotation[:]:
            newSubBbox = ResizeBbox(item['bbox_sub'],originalSize, resize)
            newObjBbox = ResizeBbox(item['bbox_obj'],originalSize, resize)

            subBbox.append(newSubBbox)
            objBbox.append(newObjBbox)

            sub.append(item['id_sub'])
            obj.append(item['id_obj'])

            attributeSub.append(item['attr_sub_id'])
            attributeObj.append(item['attr_obj_id'])

            rel.append(item['rel_id'])

        target = {
            'subBbox': torch.tensor(subBbox),
            'objBbox': torch.tensor(objBbox),
            'sub': torch.tensor(sub),
            'obj': torch.tensor(obj),
            'attributeSub': torch.tensor(attributeSub),
            'attributeObj': torch.tensor(attributeObj),
            'rel': torch.tensor(rel)
        }

        return imageTransform, target

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # chuyển từ tensor sang numpy
    img = np.clip(img, 0,1)
    plt.imshow(img)
    plt.axis('off')  # không hiển thị trục

def ResizeBbox(bbox, inSize, outSize):
    """
    Thay đổi kích thước của bounding box dựa trên tỷ lệ thay đổi kích thước của hình ảnh.

    Args:
    - bbox (list): Bounding box gốc với định dạng [xmin, ymin, width, height].
    - in_size (tuple): Kích thước gốc của hình ảnh (width, height).
    - out_size (tuple): Kích thước mới của hình ảnh sau khi resize (width, height).

    Returns:
    - list: Bounding box mới đã được thay đổi kích thước.
    """
    xmin, ymin, width, height = bbox
    x_scale = outSize[0] / inSize[0]
    y_scale = outSize[1] / inSize[1]

    new_xmin = xmin * x_scale
    new_ymin = ymin * y_scale
    new_width = width * x_scale
    new_height = height * y_scale

    return [new_xmin, new_ymin, new_width, new_height]

def GetAllData(imageDir, annoDir):
    imageList = []
    annoList = []
    for item in os.listdir(imageDir):
        #imageList.append(os.path.join(vgRoot + vgImg, item))
        imageList.append(imageDir + '/' + item)
        annoList.append(annoDir + '/' + item.replace('.jpg', '.json'))
    return imageList, annoList

def CheckSample(idx, dataset):
    #print(dataset.__len__())
    imageTransform, target = dataset.__getitem__(idx)
    #print(target)
    imshow(imageTransform)
    plt.show()

def BuildDataset(mode):
    dataset = None
    if(mode == 'train'):
        imgDir = vgRoot + vgImageTrain
        annDir = vgRoot + vgAnnoTrain
        dataList = GetAllData(imageDir=imgDir, annoDir=annDir)

        dataset = DatasetLoader(dataList, transform=MyTransform(resize, mean, std), mode=mode)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch, drop_last=True)

        datasetLoader = DataLoader(dataset, batch_sampler=batch_sampler_train, collate_fn=ults.collate_fn)

    elif(mode == 'val'):
        imgDir = vgRoot + vgImageVal
        annDir = vgRoot + vgAnnoVal
        dataList = GetAllData(imageDir=imgDir, annoDir=annDir)
        
        dataset = DatasetLoader(dataList, transform=MyTransform(resize, mean, std), mode=mode)
        sampler_val = torch.utils.data.SequentialSampler(dataset)
        datasetLoader = DataLoader(dataset, batch_size=batch, sampler=sampler_val, drop_last=False, collate_fn=ults.collate_fn)

    return datasetLoader


if __name__=='__main__':
    # trainList = GetAllData(vgRoot+vgImg)

    # trainDataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode='train')
    # sampler_train = torch.utils.data.RandomSampler(trainDataset)
    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, batch, drop_last=True)
    # trainDataLoader = DataLoader(trainDataset, batch_sampler=batch_sampler_train, collate_fn=ults.collate_fn)

    
    trainDataLoader = BuildDataset(mode='train')
    dataloaderDict = {
        "train": trainDataLoader,
        "val": None
    }
    batchIter = iter(dataloaderDict['train'])
    inputs, target = next(batchIter)
    print(len(target))
    print(target[0])
    imshow(inputs.tensors[0])


    #print(trainDataset.__len__())
    # index = 10
    # imageTransform, target = trainDataset.__getitem__(index)
    # print(target)
    #print(inputs.tensors[0].size())
    #imshow(inputs.tensors[0])
    #imshow(imageTransform)

    plt.show()