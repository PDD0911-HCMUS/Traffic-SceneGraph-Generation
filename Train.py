import ConfigArgs as args
from Datasets.DatasetLoader_3 import build_data, imshow
from Datasets.Util import nested_tensor_from_tensor_list, get_rank, is_main_process, save_on_master, get_coco_api_from_dataset, collate_fn
import matplotlib.pyplot as plt
from ModelCreation.SGGModel import build
import torch
torch.cuda.empty_cache()
from pathlib import Path
from tqdm import tqdm
from TrainEngine import train_one_epoch, evaluate
import numpy as np
import random
import json
from torch.utils.data import DataLoader


#from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    seed = 42 + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:2')
    aux_loss = False
    dec_layers = 6
    lr_drop = 3
    ouputDir = Path(args.out_dir)
    
    print('Device is being used: ', device)
    model, criterion, postprocessors = build()
    model.to(device)
    criterion.to(device)
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

    trainData = build_data('train', args.ann_path, args.img_folder)
    validData = build_data('val', args.ann_path, args.img_folder)

    sampler_train = torch.utils.data.RandomSampler(trainData)
    sampler_val = torch.utils.data.SequentialSampler(validData)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    base_ds = get_coco_api_from_dataset(validData)

    trainDataLoader = DataLoader(trainData, batch_sampler=batch_sampler_train, collate_fn=collate_fn)
    validDataLoader = DataLoader(validData, args.batch_size, sampler=sampler_val,drop_last=False, collate_fn=collate_fn)

    print("Start training")
    for epoch in range(0,100):
        train_stats = train_one_epoch(model, criterion, trainDataLoader, optimizer, device, epoch,0.1)
        lr_scheduler.step()
        checkpoint_paths = [ouputDir / 'checkpoint.pth']
        if (epoch + 1) % lr_drop == 0 or (epoch + 1) % 2 == 0:
            checkpoint_paths.append(ouputDir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        ###
        # TODO: make evaluation in here
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, validDataLoader, base_ds, device)
        ###
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        torch.save(model.state_dict(), ouputDir / f'eval/{epoch:03}.pth')
        if (epoch + 1) % 2 == 0:
            with (ouputDir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            torch.save(model.state_dict(), ouputDir / f'eval/{epoch:03}.pth')

    torch.save(model.state_dict(), ouputDir / f'eval/{epoch:03}.pth')

    dataloaderDict = {
        "train": trainDataLoader,
        "val": validDataLoader
    }
    batchIter = iter(dataloaderDict['train'])
    inputs, target = next(batchIter)
    print(len(target))
    print(target[0])
    imshow(inputs.tensors[0])
    plt.show()


    # matcher_sub = build_matcher_sub()
    # model.train()
    # criterion.train()
    # for images, targets in tqdm(trainDataLoader):
    #     images = images.to(device)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #     outputs=model(images)
    #     # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
    #     # indices_sub = matcher_sub(outputs_without_aux, targets)
    #     loss_dict = criterion(outputs, targets)
    #     # print(indices_sub[0])
    #     print('pred_logits: ', outputs['pred_logits'].size())
    #     print('sub_logits: ', outputs['sub_logits'].size())
    #     print('obj_logits: ', outputs['obj_logits'].size())

    #     print('pred_boxes: ', outputs['pred_boxes'].size())
    #     print('sub_boxes: ', outputs['sub_boxes'].size())
    #     print('obj_boxes: ', outputs['obj_boxes'].size())
    #     # print('pred_rel: ', out['pred_rel'].size())
    #     break