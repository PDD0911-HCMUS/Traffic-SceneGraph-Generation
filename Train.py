from Datasets.DatasetLoader import GetAllImage, DatasetLoader, MyTransform
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
trainDataLoader = DataLoader(trainDataset, batch_size=batch, shuffle=True)

dataloaderDict = {
    "train": trainDataLoader,
    "val": None
}

batchIter = iter(dataloaderDict['train'])
inputs, objetcContext, attrContext, label = next(batchIter)
print(len(objetcContext))
print(len(attrContext))
print(len(label))


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
output= model(x)
print("Output shape:", output.shape)
print(output[0])
gt_bbox = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
pr_bbox = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
loss = ComputeGIoU(gt_bbox, pr_bbox, reduction='none')
print(loss)

numEpochs = 3
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Chọn optimizer

for epoch in range(numEpochs):
    model.train()
    for input,objetcContext, attrContext, target in trainDataLoader:
        optimizer.zero_grad()

        output = model(input)
        print(output)
        print(target)
        break
