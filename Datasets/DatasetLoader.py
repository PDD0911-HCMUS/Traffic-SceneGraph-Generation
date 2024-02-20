import os
import json
from typing import Any
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

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
batch = 4

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
        # print(imagePath)
        image = Image.open(imagePath)
        originalSize = image.size
        imageTransform = self.transform(image, self.mode)

        annotation = os.path.join(imagePath.replace(vgImg, vgGT).replace('.jpg', '.json'))
        annotation = open(annotation)
        annotation = json.load(annotation)
        
        sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel = [],[],[], [],[],[], [] 
        objetcContext, attrContext, label = [],[],[]
        labelCouple, labelCoupleAttr, labelRel = [],[],[]

        for item in annotation:
            newSubBbox = ResizeBbox(item['sub_bbox'],originalSize, resize)
            newObjBbox = ResizeBbox(item['obj_bbox'],originalSize, resize)

            newSubAttr = ResizeAttributeAnno(item['att_sub'])
            newObjAttr = ResizeAttributeAnno(item['att_obj'])

            # sub.append(item['id_sub'])
            # subBbox.append(newSubBbox)
            # attributeSub.append(item['att_sub'])

            # obj.append(item['id_obj'])
            # objBbox.append(newObjBbox)
            # attributeObj.append(item['att_obj'])

            # rel.append(item['rel'])

            # objetcContext += [[item['id_sub']] + newSubBbox+[item['id_obj']]+newObjBbox]
            # attrContext += [newSubBbox+newSubAttr + newObjBbox+newObjAttr]

            label += [[item['id_sub']]+newSubBbox+newSubAttr + [item['id_obj']]+newObjBbox+newObjAttr + [item['rel']]]

            labelCouple += [[item['id_sub']]+newSubBbox + [item['id_obj']]+newObjBbox]
            labelCoupleAttr += [[item['id_sub']]+newSubBbox+newSubAttr + [item['id_obj']]+newObjBbox+newObjAttr]
            labelRel += [[item['id_sub']]+newSubBbox + [item['id_obj']]+newObjBbox + [item['rel']]]

        #return imageTransform, objetcContext, attrContext, label
        return imageTransform, label, labelCouple, labelCoupleAttr, labelRel
        #return imageTransform, sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel, label

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

def CheckLenghtLabel(label):
    for item in label:
        # print(len(item))
        print(item)

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

# trainList = GetAllImage(vgRoot+vgImg)

# trainDataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode='train')
# trainDataLoader = DataLoader(trainDataset, batch_size=batch, shuffle=True)

# dataloaderDict = {
#     "train": trainDataLoader,
#     "val": None
# }

# batchIter = iter(dataloaderDict['train'])
# inputs, objetcContext, attrContext, label = next(batchIter)
# print(len(objetcContext))
# print(len(attrContext))
# print(len(label))


# print(trainDataset.__len__())
# index = 100
# # imageTransform, sub,subBbox,attributeSub, obj,objBbox,attributeObj, rel, label = trainDataset.__getitem__(index)
# imageTransform, objetcContext, attrContext, label = trainDataset.__getitem__(index)
# print(imageTransform.shape)
# # print('ID Subjects: ', sub[:2])
# # print('BBox Subjects: ', subBbox[:2])
# # print('Attribute Subjects: ', attributeSub[:2])
# # print('Relation: ', rel[:2])
# print(len(objetcContext))
# print(len(attrContext))
# print(len(label))

# print("Label: ", label[0])
# print("ObjectContext: ", objetcContext[0])
# print("AttrContext: ", attrContext[0])

# imshow(imageTransform)

# plt.show()