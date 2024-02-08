from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa mô hình của bạn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)
        return x

# Khởi tạo mô hình và TensorBoard writer
model = MyModel()
writer = SummaryWriter()

# Tạo dummy input phù hợp với input layer của mô hình
dummy_input = torch.randn(1, 1, 28, 28)

# Thêm mô hình vào TensorBoard
writer.add_graph(model, dummy_input)
writer.close()
