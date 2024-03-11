from Datasets.DatasetLoader_3 import BuildDataset, imshow
from Datasets.Util import nested_tensor_from_tensor_list, get_rank, is_main_process
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ModelCreation.ComponentsModel_relt import build
import torch.optim as optim
import torch
#from tqdm import tqdm
from TrainEngine import train_one_epoch
import numpy as np
import random
import json

#from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    seed = 42 + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr_drop = 200
    ouputDir = 'CheckPoint'

    # dataloaderDict = {
    #     "train": trainDataLoader,
    #     "val": validDataLoader
    # }

    # batchIter = iter(dataloaderDict['train'])
    # inputs, target = next(batchIter)
    # print(len(target))
    # print(target[0])
    # imshow(inputs.tensors[0])
    # plt.show()
    print('Device is being used: ', device)
    model, criterion = build(device = device)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

    trainDataLoader = BuildDataset(mode='train')
    validDataLoader = BuildDataset(mode='valid')

    print("Start training")
    for epoch in range(0,300):
        train_stats = train_one_epoch(
            model, criterion, trainDataLoader, optimizer, device, epoch,
            0.1)
        lr_scheduler.step()
        checkpoint_paths = [ouputDir / 'checkpoint.pth']
        if (epoch + 1) % lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(ouputDir / f'checkpoint{epoch:04}.pth')

        ###
        # TODO: make evaluation in here
        ###
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if ouputDir and is_main_process():
            with (ouputDir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # for images, targets in tqdm(trainDataLoader):
    #     out=model(images)
    #     # print('source size: ', src.size())
    #     # print('mask size: ',mask.size())
    #     # print(cp.size())
    #     # print(hs.size())
    #     print('pred_sub_logits: ', out['pred_sub_logits'].size())
    #     print('pred_obj_logits: ', out['pred_obj_logits'].size())
    #     print('pred_boxes_sub: ', out['pred_boxes_sub'].size())
    #     print('pred_boxes_obj: ', out['pred_boxes_obj'].size())
    #     # print('pred_rel: ', out['pred_rel'].size())
    #     # sizes = [len(v["subBbox"]) for v in targets]
    #     # print(sizes)
    #     break