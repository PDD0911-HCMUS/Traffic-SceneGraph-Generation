import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, embedding_size, support_vector_size, output_size):
        super(AttentionModule, self).__init__()
        self.support_vector = nn.Parameter(torch.randn(support_vector_size), requires_grad=True)
        self.attention_layer = nn.Linear(embedding_size + support_vector_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Giả sử x là vector embedding từ CNN backbone
        # Mở rộng support_vector để phù hợp với batch_size của x
        support_vector_batch = self.support_vector.expand(x.size(0), -1)
        # Kết hợp vector embedding và support vector
        combined_input = torch.cat((x, support_vector_batch), 1)
        attention_scores = self.attention_layer(combined_input)
        attention_weights = self.softmax(attention_scores)
        return attention_weights

# Giả sử kích thước
embedding_size = 256  # Kích thước vector embedding từ CNN backbone
support_vector_size = 10  # Kích thước của vector hỗ trợ
output_size = 1  # Đối với ví dụ này, giả sử output là một điểm số attention

# Tạo mô hình
attention_module = AttentionModule(embedding_size, support_vector_size, output_size)

# Tạo dữ liệu giả để kiểm tra
x = torch.randn(3, embedding_size)  # Giả sử batch_size là 3

# Forward pass qua module attention
attention_weights = attention_module(x)
print(attention_weights)
