from Datasets.DatasetLoader_3 import BuildDataset, imshow
from Datasets.Util import nested_tensor_from_tensor_list, get_rank, is_main_process, save_on_master
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ModelCreation.ComponentsModel_relt import build
import torch.optim as optim
import torch
torch.cuda.empty_cache()
from pathlib import Path
from tqdm import tqdm
from TrainEngine import train_one_epoch, evaluate
import numpy as np
import random
import json
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ouputDir = 'Checkpoint/'
cpkt = '049.pth'
#cpkt = 'checkpoint0032.pth'

image_test_dir = 'Datasets/VisualGenome/Train/image/'
image_test_name = '48.jpg'

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Gọi mô hình gốc và lấy đầu ra
        output = self.model(x)
        # Trả về một hoặc nhiều thành phần cụ thể từ dictionary hoặc chuyển đổi nó thành tuple hoặc NamedTuple
        return output['pred_sub_logits'] 
    
if __name__ == '__main__':
    
    seed = 42 + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    
    print('Device is being used: ', device)
    aux_loss = False
    dec_layers = 6
    model, criterion, postprocessors_sub, postprocessors_obj = build(device = device, aux_loss=aux_loss, dec_layers=dec_layers)
    state_dict = torch.load(ouputDir + cpkt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

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

    print("Start Evaluation")
    image = Image.open(image_test_dir + image_test_name)
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probas_sub = outputs['pred_sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['pred_obj_logits'].softmax(-1)[0, :, :-1]
        
        print('pred_sub_logits: ', outputs['pred_sub_logits'][0], outputs['pred_sub_logits'][0].size())
        print('pred_obj_logits: ', outputs['pred_obj_logits'][0], outputs['pred_obj_logits'][0].size())

        sub_bboxes_scaled = rescale_bboxes(outputs['pred_boxes_sub'][0], image.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['pred_boxes_obj'][0], image.size)

        print('sub: ', torch.argmax(probas_sub[0]))
        print('obj: ', torch.argmax(probas_obj[0]))
        print(sub_bboxes_scaled[0])
        print(obj_bboxes_scaled[0])

        print('pred_boxes_sub: ', sub_bboxes_scaled)

        imCopy = image.copy()
        imageDraw = ImageDraw.Draw(imCopy)
        for itemS, itemO in zip(sub_bboxes_scaled[:5], obj_bboxes_scaled[:5]):
            # itemS = sub_bboxes_scaled[0].tolist()
            # itemO = obj_bboxes_scaled[0].tolist()
            x_sub, y_sub, w_sub, h_sub = itemS[0],itemS[1],itemS[2],itemS[3]
            x_obj, y_obj, w_obj, h_obj = itemO[0],itemO[1],itemO[2],itemO[3]
            #print(x1, y1, x2, y2, item['names'][0])
            imageDraw.rectangle([(x_sub, y_sub), (w_sub + x_sub, h_sub + y_sub)], outline ="red", width=2) 
            imageDraw.rectangle([(x_obj, y_obj), (w_obj + x_obj, h_obj + y_obj)], outline ="blue", width=2) 
        # imageDraw.text((x_sub+5, y_sub+5), item['cls_sub'], fill='black')
        # imageDraw.text((x_obj+5, y_obj+5), item['cls_obj'], fill='black')
        plt.figure(figsize=(16,9))
        plt.imshow(imCopy)
        plt.show()


