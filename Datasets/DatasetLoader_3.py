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

vgAttr = vgRoot+'Annotation/attributes.json'
vgSG = vgRoot+'Annotation/scene_graphs.json'

vgExAttr = 'ExtractAttribute'
vgExRel = 'ExtractRelation'
vgImg = 'Image'
vgGT = 'GTTraffic'

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
    def __init__(self, imageList, transform = None, mode = 'train'):
        self.imageList = imageList
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, index):
        imagePath = self.imageList[index]
        # #print(imagePath)
        

        annotation = os.path.join(imagePath.replace(vgImg, vgGT).replace('.jpg', '.json'))
        annotation = open(annotation)
        annotation = json.load(annotation)

        image = Image.open(imagePath)
        originalSize = image.size
        imageTransform = self.transform(image, self.mode)
        
        sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel = [],[],[], [],[],[], [] 
        objetcContext, attrContext, label = [],[],[]
        labelCouple, labelCoupleAttr, labelRel = [],[],[]

        for item in annotation[:5]:
            newSubBbox = ResizeBbox(item['bbox_sub'],originalSize, resize)
            newObjBbox = ResizeBbox(item['bbox_obj'],originalSize, resize)

            # newSubAttr = ResizeAttributeAnno(item['attr_sub_id'])
            # newObjAttr = ResizeAttributeAnno(item['attr_obj_id'])

            subBbox.append(newSubBbox)
            objBbox.append(newObjBbox)

            sub.append(item['id_sub'])
            obj.append(item['id_obj'])

            attributeSub = item['attr_sub_id']
            attributeObj = item['attr_obj_id']

            rel.append(item['rel_id'])

            
            #label += [[item['id_sub']]+newSubBbox+newSubAttr + [item['id_obj']]+newObjBbox+newObjAttr + [item['rel']]]

            # label.append([item['id_sub']]+newSubBbox+newSubAttr + [item['id_obj']]+newObjBbox+newObjAttr + [item['rel_id']])
            # labelCouple.append([item['id_sub']]+newSubBbox + [item['id_obj']]+newObjBbox)
            # labelCoupleAttr.append([item['id_sub']]+newSubBbox+newSubAttr + [item['id_obj']]+newObjBbox+newObjAttr)
            # labelRel.append([item['id_sub']]+newSubBbox + [item['id_obj']]+newObjBbox + [item['rel_id']])

            # labelCouple += [[item['id_sub']]+newSubBbox + [item['id_obj']]+newObjBbox]
            # labelCoupleAttr += [[item['id_sub']]+newSubBbox+newSubAttr + [item['id_obj']]+newObjBbox+newObjAttr]
            # labelRel += [[item['id_sub']]+newSubBbox + [item['id_obj']]+newObjBbox + [item['rel']]]

        # target = {
        #     'annotation': torch.tensor(label),
        #     'labelCouple': torch.tensor(labelCouple),
        #     'labelCoupleAttr': torch.tensor(labelCoupleAttr),
        #     'labelRel': torch.tensor(labelRel)
        # }

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

def ResizeAttributeAnno(attr: list):
    """
    Giá trị mặc định của None là 50852
    """
    if(len(attr) < 2):
        attr.append(50852)
        ResizeAttributeAnno(attr)
        return attr
    else:
        return attr[:2]

# def CheckLenghtLabel(label):
#     for item in label:
#         # #print(len(item))
#         #print(item)

def GetAllImage(imageDir):
    imageList = []
    for item in os.listdir(imageDir):
        #imageList.append(os.path.join(vgRoot + vgImg, item))
        imageList.append(vgRoot + vgImg + '/' + item)
    return imageList[:]

def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0]) #sample[0]=img
        targets.append(torch.FloatTensor(sample[1])) # sample[1]=annotation
    #[3, 300, 300]
    # (batch_size, 3, 300, 300)
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets

def CheckSample(idx, dataset):
    #print(dataset.__len__())
    imageTransform, target = dataset.__getitem__(idx)
    #print(target)
    imshow(imageTransform)
    plt.show()

def BuildDataset(imPath, mode):
    trainList = GetAllImage(imPath)
    dataset = None
    if(mode == 'train'):
        dataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode=mode)
    elif(mode == 'val'):
        dataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode=mode)
    return dataset


if __name__=='__main__':
    trainList = GetAllImage(vgRoot+vgImg)

    trainDataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode='train')
    trainDataLoader = DataLoader(trainDataset, batch_size=batch, collate_fn=ults.collate_fn , shuffle=True)

    dataloaderDict = {
        "train": trainDataLoader,
        "val": None
    }

    # batchIter = iter(dataloaderDict['train'])
    # inputs, target = next(batchIter)
    # #print(len(target['annotation']))


    #print(trainDataset.__len__())
    index = 10
    imageTransform, target = trainDataset.__getitem__(index)
    print(target)
    #print(inputs.tensors[0].size())
    #imshow(inputs.tensors[0])
    imshow(imageTransform)

    plt.show()