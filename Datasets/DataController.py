import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

vgRoot = 'Datasets/VisualGenome/'

vgAttr = vgRoot+'Annotation/attributes.json'
vgSG = vgRoot+'Annotation/scene_graphs.json'

vgExAttr = 'ExtractAttribute'
vgExRel = 'ExtractRelation'
vgGT = 'ExtractGT'

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.img_names = [file_name for file_name in os.listdir(img_dir) if file_name.endswith('.jpg')]
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.json'))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        # Trích xuất thông tin cần thiết từ annotation
        id_subs, att_subs, sub_bboxes, id_objs, att_objs, obj_bboxes, rels = [], [], [], [], [], [], []
        for item in annotation:
            id_subs.append(item['id_sub'])
            att_subs.append(item['att_sub'][0])
            sub_bboxes.append(item['sub_bbox'])
            id_objs.append(item['id_obj'])
            att_objs.append(item['att_obj'][0])
            obj_bboxes.append(item['obj_bbox'])
            rels.append(item['rel'])
        
        if self.transform:
            image = self.transform(image)
        
        # Trả về hình ảnh và các thông tin riêng biệt
        return image, id_subs, att_subs, sub_bboxes, id_objs, att_objs, obj_bboxes, rels

# Biến đổi cho hình ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Khởi tạo dataset
dataset = CustomDataset(img_dir=os.path.join(vgRoot, 'ExtractImage'), annotation_dir=os.path.join(vgRoot, vgGT), transform=transform)

# Khởi tạo DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Hàm để hiển thị một hình ảnh
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # chuyển từ tensor sang numpy
    plt.imshow(img)
    plt.axis('off')  # không hiển thị trục
    

# Lấy một batch của dữ liệu
images, id_subs, att_subs, sub_bboxes, id_objs, att_objs, obj_bboxes, rels = next(iter(data_loader))

# Hiển thị một số hình ảnh và thông tin tương ứng
fig = plt.figure(figsize=(16, 9))
for i in range(len(images)):
    print(i)
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    imshow(images[i])
    #ax.set_title(f'ID Sub: {id_subs[i]}\nID Obj: {id_objs[i]}')
    # Thêm bất kỳ thông tin nào bạn muốn hiển thị cùng hình ảnh

plt.show()
