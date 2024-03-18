import os
import json
from typing import Any
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import Datasets.Util as ults
import Datasets.TransformUtils as T
# import Util as ults
# import TransformUtils as T
from pycocotools import mask as coco_mask
from tqdm import tqdm

vgRoot = 'Datasets/VisualGenome/'

vgImageTrain = 'Train/image'
vgAnnoTrain = 'Train/target'

vgImageVal = 'Val/image'
vgAnnoVal = 'Val/target'
batch = 2

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.NormalizeSO([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlipSO(),
            T.RandomSelect(
                T.RandomResizeSO(scales, max_size=1333),
                T.Compose([
                    #T.RandomResizeSO([400, 500, 600]),
                    #T.RandomResizeSO(scales, max_size=1333),
                    T.RandomResizeSO([400, 500]),
                    T.RandomResizeSO(scales, max_size=512)
                ])
            ),
            normalize])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResizeSO([800], max_size=1333),
            T.RandomResizeSO([512], max_size=512),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class DatasetLoader(Dataset):
    def __init__(self, imageList, transform = None, mode = 'train'):
        self.imageList = imageList
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, index):
        imagePath = self.imageList[index]
        # print(imagePath)
        
        if(self.mode == 'train'):
            target = open(os.path.join(imagePath.replace(vgImageTrain, vgAnnoTrain).replace('.jpg', '.json')))
            target = json.load(target)
        if(self.mode == 'val'):
            target = open(os.path.join(imagePath.replace(vgImageVal, vgAnnoVal).replace('.jpg', '.json')))
            target = json.load(target)
            
        image = Image.open(imagePath)
        for item in target.keys():
            target[item] = torch.as_tensor(target[item])

        _, unique_indices_sub = target['subBbox'].unique(dim=0, return_inverse=True)
        unique_subBbox = target['subBbox'][unique_indices_sub]
        unique_sub = target['sub'][unique_indices_sub]

        _, unique_indices_obj = target['objBbox'].unique(dim=0, return_inverse=True)
        unique_objBbox = target['objBbox'][unique_indices_obj]
        unique_obj = target['obj'][unique_indices_obj]

        target_main = {
            'subBbox': unique_subBbox,
            'objBbox': unique_objBbox,
            'sub': unique_sub,
            'obj': unique_obj,
            'image_id': target['image_id'],
            'orig_size': target['orig_size']
        }
        image, target = self.transform(image, target_main)

        return image, target

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # chuyển từ tensor sang numpy
    img = np.clip(img, 0,1)
    plt.imshow(img)
    plt.axis('off')  # không hiển thị trục

def GetAllData(imageDir):
    imageList = []
    for item in os.listdir(imageDir):
        #imageList.append(os.path.join(vgRoot + vgImg, item))
        imageList.append(imageDir + '/' + item)
    return imageList

def BuildDataset(mode):
    dataset = None
    datasetLoader = None
    if(mode == 'train'):
        imgDir = vgRoot + vgImageTrain
        dataList = GetAllData(imageDir=imgDir)

        dataset = DatasetLoader(dataList[:30000], transform=make_coco_transforms(image_set=mode), mode=mode)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch, drop_last=True)
        datasetLoader = DataLoader(dataset, batch_sampler=batch_sampler_train, collate_fn=ults.collate_fn)

    elif(mode == 'val'):
        imgDir = vgRoot + vgImageVal
        dataList = GetAllData(imageDir=imgDir)
        
        dataset = DatasetLoader(dataList[:5000], transform=make_coco_transforms(image_set=mode), mode=mode)
        sampler_val = torch.utils.data.SequentialSampler(dataset)
        datasetLoader = DataLoader(dataset, batch_size=batch, sampler=sampler_val, drop_last=False, collate_fn=ults.collate_fn)

    return datasetLoader

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target

        boxes_sub = anno["subBbox"] 
        boxes_obj = anno["objBbox"]
        # guard against no boxes via resizing
        boxes_sub = torch.as_tensor(boxes_sub, dtype=torch.float32).reshape(-1, 4)
        boxes_sub[:, 2:] += boxes_sub[:, :2]
        boxes_sub[:, 0::2].clamp_(min=0, max=w)
        boxes_sub[:, 1::2].clamp_(min=0, max=h)

        boxes_obj = torch.as_tensor(boxes_obj, dtype=torch.float32).reshape(-1, 4)
        boxes_obj[:, 2:] += boxes_obj[:, :2]
        boxes_obj[:, 0::2].clamp_(min=0, max=w)
        boxes_obj[:, 1::2].clamp_(min=0, max=h)

        classes_sub = anno["sub"]
        classes_sub = torch.tensor(classes_sub, dtype=torch.int64)

        classes_obj = anno["obj"]
        classes_obj = torch.tensor(classes_obj, dtype=torch.int64)

        target = {}
        target["subBbox"] = boxes_sub
        target["objBbox"] = boxes_obj
        target["sub"] = classes_sub
        target["obj"] = classes_obj
        target["image_id"] = image_id
        target["rel"] = anno['rel']

        # for conversion to coco api

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


# if __name__=='__main__':

#     trainDataLoader = BuildDataset(mode='train')
#     validDataLoader = BuildDataset(mode='val')
#     # dataloaderDict = {
#     #     "train": trainDataLoader,
#     #     "val": validDataLoader
#     # }
#     # batchIter = iter(dataloaderDict['train'])
#     # inputs, target = next(batchIter)
#     # print(len(target))
#     # print(target[0])
#     # imshow(inputs.tensors[0])
#     # plt.show()
#     t = 0
#     for images, targets in tqdm(trainDataLoader):
#         # out=model(images)
#         # # print('source size: ', src.size())
#         # # print('mask size: ',mask.size())
#         # # print(cp.size())
#         # # print(hs.size())
#         t = t + 1