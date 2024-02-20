import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b7')

    def forward(self, x):
        features = self.cnn.extract_features(x)
        global_features = F.adaptive_avg_pool2d(features, 1)
        global_features = global_features.view(global_features.size(0), -1)
        return global_features

class AttentionLayer(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(AttentionLayer, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        
        # Khởi tạo ma trận trọng số cho W^Q, W^K, W^V
        self.W_q = torch.nn.Linear(d_model, d_k * h, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_k * h, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_v * h, bias=False)

        # Linear layer để kết hợp các đầu ra từ các heads
        self.linear = torch.nn.Linear(d_v * h, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Biến đổi x qua W^Q, W^K, W^V
        Q = self.W_q(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # [batch_size, h, seq_len, d_k]
        K = self.W_k(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)
        
        # Tính attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        
        # Áp dụng attention lên V
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)
        
        # Kết hợp các heads
        output = self.linear(context)
        
        return output, attn

class OutputCNN(nn.Module):
    def __init__(self):
        super(OutputCNN, self).__init__()
        # Định nghĩa các lớp Conv1d
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Lớp Fully Connected để chuyển từ đặc trưng học được sang output 18 chiều
        self.fc = nn.Linear(32 * (40 // 4), 18*15)  # Giả định sau 2 lần pooling, kích thước giảm 1 nửa mỗi lần
        
    def forward(self, x):
        # Xử lý qua Conv1d và MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten đầu ra để sử dụng trong lớp Fully Connected
        x = x.view(x.size(0), -1)  # x.size(0) là batch size
        
        # Đưa qua lớp Fully Connected để nhận vector output 18*15 chiều
        x = self.fc(x)

        # x = divide_chunks(x, n=18)
        # sub = [x[0]]
        # sub_box = x[1:5]
        # sub_att = x[5:7]

        # obj = [x[7]]
        # obj_bbox = x[8:12]
        # obj_att = x[12:14]

        # rel = [x[-1]]
        return x

class OutputCoor(nn.Module):
    def __init__(self):
        super(OutputCoor, self).__init__()
        # Định nghĩa các lớp Conv1d
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Lớp Fully Connected để chuyển từ đặc trưng học được sang output 18 chiều
        self.fc = nn.Linear(32 * (40 // 4), 18*5*2)  # Giả định sau 2 lần pooling, kích thước giảm 1 nửa mỗi lần
        
    def forward(self, x):
        # Xử lý qua Conv1d và MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten đầu ra để sử dụng trong lớp Fully Connected
        x = x.view(x.size(0), -1)  # x.size(0) là batch size
        
        # Đưa qua lớp Fully Connected để nhận vector output 18*15 chiều
        x = self.fc(x)

        return x
    
class OutputAttr(nn.Module):
    def __init__(self):
        super(OutputAttr, self).__init__()
        # Định nghĩa các lớp Conv1d
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Lớp Fully Connected để chuyển từ đặc trưng học được sang output 18 chiều
        self.fc = nn.Linear(32 * (40 // 4), 18*8*4)  # Giả định sau 2 lần pooling, kích thước giảm 1 nửa mỗi lần
        
    def forward(self, x):
        # Xử lý qua Conv1d và MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten đầu ra để sử dụng trong lớp Fully Connected
        x = x.view(x.size(0), -1)  # x.size(0) là batch size
        
        # Đưa qua lớp Fully Connected để nhận vector output 18*15 chiều
        x = self.fc(x)

        return x
    
class OutputAttr(nn.Module):
    def __init__(self):
        super(OutputAttr, self).__init__()
        # Định nghĩa các lớp Conv1d
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Lớp Fully Connected để chuyển từ đặc trưng học được sang output 18 chiều
        self.fc = nn.Linear(32 * (40 // 4), 18*8*8)  # Giả định sau 2 lần pooling, kích thước giảm 1 nửa mỗi lần
        
    def forward(self, x):
        # Xử lý qua Conv1d và MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten đầu ra để sử dụng trong lớp Fully Connected
        x = x.view(x.size(0), -1)  # x.size(0) là batch size
        
        # Đưa qua lớp Fully Connected để nhận vector output 18*15 chiều
        x = self.fc(x)

        return x
class MyModel(nn.Module):
    def __init__(self, cnnBackbone, attentionLayerEOA, attentionLayerEAA, attentionLayerERA, outputCoor, outputAttr, outputRel):
        super(MyModel, self).__init__()
        # d_model = 512
        # d_k = d_v = 64
        # h = 8

        self.cnnBackbone = cnnBackbone
        self.attentionLayerEOA = attentionLayerEOA
        self.attentionLayerEAA = attentionLayerEAA
        self.attentionLayerERA = attentionLayerERA

        self.outputCoor = outputCoor
        self.outputAttr = outputAttr
        self.outputRel = outputRel
    
    def forward(self, x):

        image_embedding = self.cnnBackbone(x)

        q_cp = torch.randn(image_embedding.size())
        q_attr = torch.randn(image_embedding.size())

        image_embedding_cp = image_embedding + q_cp
        image_embedding_attr = image_embedding_cp + q_attr

        seq_length = 40  # Số lượng phần tử trong chuỗi
        feature_dim = image_embedding.shape[1] // seq_length  # Kích thước của mỗi phần tử chuỗi
        # Reshape tensor để có kích thước [batch_size, seq_length, feature_dim]
        input_tensor_reshaped = image_embedding.view(-1, seq_length, feature_dim)
        #print('input_tensor_reshaped: ', input_tensor_reshaped.shape)

        #Xu ly Attribute
        outputAttr, attn_weightsAttr = self.attentionLayerEOA(input_tensor_reshaped)
        #Xu ly Relation
        outputRel, attn_weightsRel = self.attentionLayerEAA(input_tensor_reshaped)
        #Xu ly dau ra
        outConcat = torch.cat((outputAttr, outputRel),-1)

        outputTar, attn_weightsTar = self.attentionLayerERA(outConcat)

        outputTar = outputTar.permute(0, 2, 1)

        output = self.outputCNN(outputTar)

        output = [item.unfold(dimension = 0,size = 15, step = 15) for item in output]
        #output = divide_chunks(output, n=15)

        return output

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



# if __name__ == '__main__':
#     # # Giả sử d_model = 512, d_k = d_v = 64, và h = 8
#     d_model = 2560//40 #old = 512
#     d = 128
#     d_k = d_v = 64
#     h = 8

    # # Khởi tạo mô hình
    # model = AttentionLayer(d_model, d_k, d_v, h)

    # # Tạo một batch dữ liệu đầu vào ngẫu nhiên
    # x = torch.rand(8, 40, 2560//40)  # 5 là batch_size, 10 là seq_len

    # # Chạy mô hình
    # output, attn_weights = model(x)

    # print(x)
    # print(output)
    # print("Output shape:", output.shape)
    # print("Attention Weights shape:", attn_weights.shape)
        
    # x = torch.randn(8, 3, 224, 224)

    # y = torch.randn(8,40,128)

    # attentionLayer = AttentionLayer(d_model, d_k, d_v, h)
    # attentionLayerEOA = attentionLayer
    # attentionLayerEAA = attentionLayer
    # attentionLayerERA = AttentionLayer(d, d_k, d_v, h)
    # cnnBackbone = CNNBackbone()
    # cnnOutput = OutputCNN()
    # #cnnOutput = SequenceModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
    # model = MyModel(cnnBackbone,attentionLayerEOA,attentionLayerEAA,attentionLayerERA, cnnOutput)
    # output= model(x)
    # print(output)
    # print(len(output))
    # print("Output shape:", output[0].shape)

    # # print("Output shape y:", y.shape)
    # print(output[0])
    # print(y[0])
    # gt_bbox = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    # pr_bbox = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    # loss = ComputeGIoU(gt_bbox, pr_bbox, reduction='none')
    # print(loss)
    # print("Attention Weights shape:", attn_weights.shape)