seed = 42

'''Args Data'''
vgRoot = 'Datasets/VisualGenome/'

vgImageTrain = 'Train/image'
vgAnnoTrain = 'Train/anno'

vgImageVal = 'Val/image'
vgAnnoVal = 'Val/anno'

resize = (224,224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch = 2

'''Args Model'''
train_backbone = 1e-5
return_interm_layers = True
backbone = 'resnet50'

nhead=8
num_encoder_layers=6
num_decoder_layers=6
dim_feedforward=2048
dropout=0.1
activation="relu"
normalize_before=False
return_intermediate_dec=True

hidden_dim = 256
num_couple = 100
num_att=100
num_classes=181
num_rel = 51

''''''
loss_ce = 1
loss_bbox =  5
loss_giou = 2

cost_class=1
cost_bbox=5
cost_giou=2

eos_coef=0.1