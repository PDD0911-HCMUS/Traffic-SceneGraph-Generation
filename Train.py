from Datasets.DatasetLoader import GetAllImage, DatasetLoader, MyTransform, my_collate_fn
from torch.utils.data import Dataset, DataLoader
from ModelCreation.ComponentsModel import *
import torch.optim as optim

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
batch = 8

trainList = GetAllImage(vgRoot+vgImg)

trainDataset = DatasetLoader(trainList, transform=MyTransform(resize, mean, std), mode='train')
trainDataLoader = DataLoader(trainDataset, batch_size=batch, shuffle=True, collate_fn=my_collate_fn)

dataloaderDict = {
    "train": trainDataLoader,
    "val": None
}

batchIter = iter(dataloaderDict['train'])
# inputs, objetcContext, attrContext, label = next(batchIter)
inputs, label = next(batchIter)
print(inputs.size()) 
print(len(label))
print(label[0].size()) # sub, sub_box, sub_att, obj, obj_box, obj_att, rel
print(label[0])


d_model = 2560//40 #old = 512
d = 128
d_k = d_v = 64
h = 8

x = torch.randn(8, 3, 224, 224)

attentionLayer = AttentionLayer(d_model, d_k, d_v, h)
attentionLayerEOA = attentionLayer
attentionLayerEAA = attentionLayer
attentionLayerERA = AttentionLayer(d, d_k, d_v, h)
cnnBackbone = CNNBackbone()
cnnOutput = OutputCNN()
model = MyModel(cnnBackbone,attentionLayerEOA,attentionLayerEAA,attentionLayerERA, cnnOutput)
# output= model(x)
# print("Output shape:", output.shape)
# print(output[0])
# gt_bbox = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
# pr_bbox = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
# loss = ComputeGIoU(gt_bbox, pr_bbox, reduction='none')
# print(loss)

numEpochs = 1
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Chọn optimizer

# for epoch in range(numEpochs):
#     model.train()
#     for input,target in trainDataLoader:
#         optimizer.zero_grad()

#         output = model(input)
#         print('output:', output)
#         print('size output: ', output.size())
#         # print('subject pred: ', output[0])
#         # print('subject bbox: ', output[1:4])
#         # print('subject attributes: ', output[5:9])
#         # print('object pred: ', output[11:14])
#         # print('object bbox: ', output[0])
#         # print('object attributes: ', output[0])

#         print('target: ',target)
#         print('size target: ', target.shape)
#         break
