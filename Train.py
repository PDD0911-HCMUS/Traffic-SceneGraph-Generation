from Datasets.DatasetLoader_3 import BuildDataset, imshow
from Datasets.Util import nested_tensor_from_tensor_list
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ModelCreation.ComponentsModel_relt import build
import torch.optim as optim
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    trainDataLoader = BuildDataset(mode='train')
    #validDataLoader = BuildDataset(mode='valid')
    # dataloaderDict = {
    #     "train": trainDataLoader,
    #     "val": validDataLoader
    # }

    model = build()
    for images, targets in tqdm(trainDataLoader):
        out=model(images)
        # print('source size: ', src.size())
        # print('mask size: ',mask.size())
        # print(cp.size())
        # print(hs.size())
        print(out['pred_logits'].size())
        print(out['pred_boxes'].size())
        # query_embed = cp.unsqueeze(1).repeat(1, 2, 1)
        # tgt = torch.zeros_like(query_embed)
        # print(query_embed[0])
        # print(tgt.size())
        break