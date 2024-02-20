import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b7')

    def forward(self, x):
        features = self.cnn.extract_features(x)
        global_features = F.adaptive_avg_pool2d(features, 1)
        global_features = global_features.view(global_features.size(0), -1)
        return global_features

class EsitmateAttributeAttentionLayer(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(EsitmateAttributeAttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo ma trận trọng số cho Q, K, V
        self.W_q = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.q = nn.Linear(feature_dim, hidden_dim, bias=False)
        
    def forward(self, x):
        # Biến đổi x qua W_q, W_k, W_v
        Q = self.W_q(x)
        K = self.W_k(x)
        q = self.q(x)
        V = self.W_v(x) + q
        
        # Tính điểm attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Áp dụng attention lên V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
class EstimateCoupleAttentionLayer(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(EstimateCoupleAttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo ma trận trọng số cho Q, K, V
        self.W_q = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.q = nn.Linear(feature_dim, hidden_dim, bias=False)
        
    def forward(self, x):
        # Biến đổi x qua W_q, W_k, W_v
        Q = self.W_q(x)
        K = self.W_k(x)
        q = self.q(x)
        V = self.W_v(x) + q
        
        # Tính điểm attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Áp dụng attention lên V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class FFNCoupleRegr(nn.Module):
    def __init__(self):
        super(FFNCoupleRegr, self).__init__()
        # Định nghĩa các lớp Conv1d
        self.fc1 = nn.Linear(1024, 512)  # Giảm kích thước xuống
        self.fc2 = nn.Linear(512, 256)   # Tiếp tục giảm kích thước
        self.fc3 = nn.Linear(256, 180)   # 180 đơn vị đầu ra cho 18 cặp bounding box
        
    def forward(self, x):
        # Truyền qua các tầng Fully Connected với hàm kích hoạt ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Không sử dụng hàm kích hoạt ở tầng cuối cùng

        return x
class MyModel(nn.Module):
    def __init__(self, cnnBackbone, attentionLayerEOA, attentionLayerEAA, ffnCp):
        super(MyModel, self).__init__()

        self.cnnBackbone = cnnBackbone
        self.attentionLayerEOA = attentionLayerEOA
        self.attentionLayerEAA = attentionLayerEAA

        self.ffnCp = ffnCp
    
    def forward(self, x):

        image_embedding = self.cnnBackbone(x)

        seq_length = 40  # Số lượng phần tử trong chuỗi
        feature_dim = image_embedding.shape[1] // seq_length  # Kích thước của mỗi phần tử chuỗi
        # Reshape tensor để có kích thước [batch_size, seq_length, feature_dim]
        input_tensor_reshaped = image_embedding.view(-1, seq_length, feature_dim)
        #print('input_tensor_reshaped: ', input_tensor_reshaped.shape)

        #Xu ly Attribute
        outputAttr, attn_weightsAttr = self.attentionLayerEOA(image_embedding)
        #Xu ly Relation
        outputRel, attn_weightsRel = self.attentionLayerEAA(image_embedding)
        #Xu ly dau ra
        outConcat = torch.cat((outputAttr, outputRel),-1)

        bboxCp = self.ffnCp(outConcat)

        return bboxCp

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def ComputeGIoU(pred, grt, reduction='mean'):
    """
    grt: tensor (-1, 4) xyxy
    pred: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (grt[:, 2]-grt[:, 0])*(grt[:, 3]-grt[:, 1])
    pr_area = (pred[:, 2]-pred[:, 0])*(pred[:, 3]-pred[:, 1])

    # iou
    lt = torch.max(grt[:, :2], pred[:, :2])
    rb = torch.min(grt[:, 2:], pred[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    # enclosure
    lt = torch.min(grt[:, :2], pred[:, :2])
    rb = torch.max(grt[:, 2:], pred[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure-union)/enclosure
    loss = 1. - giou
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    return loss

def original_giou_loss(pred_boxes, target_boxes):
    """
    Tính GIoU loss giữa hai tập hợp của bounding boxes
    :param pred_boxes: Tensor của các predicted bounding boxes, kích thước [batch_size, 4]
    :param target_boxes: Tensor của các target bounding boxes, kích thước [batch_size, 4]
    :return: GIoU loss
    """
    # Tính toán giao diện (intersection)
    xA = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    yA = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    xB = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    yB = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    
    # Tính toán diện tích của mỗi hộp
    boxAArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    boxBArea = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    
    # Tính toán hợp (union)
    unionArea = boxAArea + boxBArea - interArea
    
    # Tính IoU
    iou = interArea / unionArea
    
    # Tính toán kích thước của hộp bao chứa cả hai hộp
    enclosing_xA = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_yA = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_xB = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_yB = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enclosingArea = (enclosing_xB - enclosing_xA) * (enclosing_yB - enclosing_yA)
    
    # Tính GIoU
    giou = iou - (enclosingArea - unionArea) / enclosingArea
    giou_loss = 1 - giou  # GIoU loss
    
    return giou_loss.mean()

def giou_loss(pred_boxes, target_boxes):
    """
    Tính GIoU loss giữa hai tập hợp của bounding boxes có định dạng [x, y, w, h]
    :param pred_boxes: Tensor của các predicted bounding boxes, kích thước [batch_size, 4]
    :param target_boxes: Tensor của các target bounding boxes, kích thước [batch_size, 4]
    :return: GIoU loss
    """
    # Chuyển đổi pred_boxes và target_boxes từ [x, y, w, h] sang [x1, y1, x2, y2]
    pred_boxes_converted = torch.cat((pred_boxes[:, :2], pred_boxes[:, :2] + pred_boxes[:, 2:]), dim=1)
    target_boxes_converted = torch.cat((target_boxes[:, :2], target_boxes[:, :2] + target_boxes[:, 2:]), dim=1)
    
    # Sử dụng pred_boxes_converted và target_boxes_converted với hàm giou_loss đã định nghĩa trước đó
    return original_giou_loss(pred_boxes_converted, target_boxes_converted)



if __name__ == '__main__':
    x = torch.randn(8, 3, 224, 224)

    # Try CNN Backbone
    cnnBackbone = CNNBackbone()
    outCNN = cnnBackbone(x)
    print("output CNN: ", outCNN.size())

    #Try EstimateCoupleAttentionLayer
    feature_dim = outCNN.size(1)
    hidden_dim = 512

    attCplayer = EstimateCoupleAttentionLayer(feature_dim, hidden_dim)
    outputCpAtt, attention_weights = attCplayer(outCNN)
    print("output Couple Attention: ", outputCpAtt.size())
    #print(outputCpAtt)

    #Try EsitmateAttributeAttentionLayer
    feature_dim = outCNN.size(1)
    hidden_dim = 512

    attAttlayer = EsitmateAttributeAttentionLayer(feature_dim, hidden_dim)
    outputAttAtt, attention_weights = attAttlayer(outCNN)
    print("output Attribute Attention: ", outputAttAtt.size())
    #print(outputAttAtt)

    #Try FFN BBox for couple
    outConcat = torch.cat((outputCpAtt, outputAttAtt),-1)
    print("outConcat: ",outConcat.size())
    ffnCp = FFNCoupleRegr()
    bboxCp = ffnCp(outConcat)
    print(bboxCp.size())
    print(attention_weights[0])

    # attention_map = attention_weights[0].cpu().detach().numpy()  # Chuyển tensor sang numpy array

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(attention_map, cmap='viridis', annot=True)
    # plt.title("Attention Weights Heatmap")
    # plt.xlabel("Key Positions")
    # plt.ylabel("Query Positions")
    # plt.show()