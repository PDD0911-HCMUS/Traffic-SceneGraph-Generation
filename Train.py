from Datasets.DatasetLoader import GetAllImage, DatasetLoader, MyTransform, my_collate_fn
from torch.utils.data import Dataset, DataLoader
from ModelCreation.ComponentsModel import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
batch = 2

trainList = GetAllImage(vgRoot+vgImg)

trainDataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode='train')
trainDataLoader = DataLoader(trainDataset, batch_size=batch, shuffle=True, collate_fn=my_collate_fn)

dataloaderDict = {
    "train": trainDataLoader,
    "val": None
}

