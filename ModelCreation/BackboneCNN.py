import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b7')

    def forward(self, x):
        features = self.cnn.extract_features(x)
        global_features = F.adaptive_avg_pool2d(features, 1)
        global_features = global_features.view(global_features.size(0), -1)
        return global_features

class EstimateObjectAttention(nn.Module):
    def __init__(self, feature_size, support_vector_size):
        super(EstimateObjectAttention, self).__init__()
        self.support_vector = nn.Parameter(torch.randn(support_vector_size), requires_grad=True)
        self.attention_layer = nn.Linear(feature_size + support_vector_size, 1)

    def forward(self, image_embedding):
        batch_size = image_embedding.size(0)
        support_vector_batch = self.support_vector.expand(batch_size, -1)
        combined_input = torch.cat((image_embedding, support_vector_batch), dim=1)
        attention_score = self.attention_layer(combined_input)
        return attention_score

class EstimateAttributeAttention(nn.Module):
    def __init__(self, feature_size, support_vector_size):
        super(EstimateAttributeAttention, self).__init__()
        self.support_vector = nn.Parameter(torch.randn(support_vector_size), requires_grad=True)
        self.attention_layer = nn.Linear(feature_size + support_vector_size, 1)

    def forward(self, image_embedding):
        batch_size = image_embedding.size(0)
        support_vector_batch = self.support_vector.expand(batch_size, -1)
        combined_input = torch.cat((image_embedding, support_vector_batch), dim=1)
        attention_score = self.attention_layer(combined_input)
        return attention_score

class EstimateRelationAttention(nn.Module):
    def __init__(self):
        super(EstimateRelationAttention, self).__init__()
        self.relation_layer = nn.Linear(2, 1)  # Assuming each attention score is a scalar

    def forward(self, object_attention, attribute_attention):
        combined_attention = torch.cat((object_attention, attribute_attention), dim=1)
        relation_score = self.relation_layer(combined_attention)
        return relation_score

class MyCompleteModel(nn.Module):
    def __init__(self):
        super(MyCompleteModel, self).__init__()
        self.cnn_backbone = CNNBackbone()
        feature_size = self.cnn_backbone.cnn._fc.in_features  # Get the feature size from EfficientNet
        self.eoa = EstimateObjectAttention(feature_size, support_vector_size=10)
        self.eaa = EstimateAttributeAttention(feature_size, support_vector_size=10)
        self.era = EstimateRelationAttention()

    def forward(self, x):
        image_embedding = self.cnn_backbone(x)
        
        object_attention = self.eoa(image_embedding)
        attribute_attention = self.eaa(image_embedding)
        
        relation_attention = self.era(object_attention, attribute_attention)
        
        return relation_attention

# Initialize the complete model
model = MyCompleteModel()
writer = SummaryWriter()

# Tạo dummy input phù hợp với input layer của mô hình
dummy_input = torch.randn(1, 3, 600, 600)

# Thêm mô hình vào TensorBoard
writer.add_graph(model, dummy_input)
writer.close()
# Example forward pass with random data
# x = torch.randn(1, 3, 600, 600)  # Example image tensor (batch size, channels, height, width)
# relation_attention_scores = model(x)
# print(relation_attention_scores)
