import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ==========================================
# 1. ĐỊNH NGHĨA MODEL (Phải khớp với file train)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

class ModernCNN(nn.Module):
    def __init__(self, num_classes=13, dropout_fc=0.5):
        super(ModernCNN, self).__init__()
        self.layer1 = ResidualBlock(1, 32, stride=1, dropout=0.1)
        self.layer2 = ResidualBlock(32, 64, stride=2, dropout=0.1)
        self.layer3 = ResidualBlock(64, 128, stride=2, dropout=0.2)
        self.layer4 = ResidualBlock(128, 256, stride=2, dropout=0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        return x
        
# ==========================================
# 2. CÁC HÀM XỬ LÝ (Giữ nguyên như cũ)
# ==========================================
def preprocess_image_from_array(img_array, target_size=28, box_size=20):
    """Xử lý ảnh đầu vào qua 8 bước"""
    steps_images = {}
    
    # 1. Original & 2. Gray
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        steps_images['1_Original'] = img_array
    else:
        gray = img_array
        steps_images['1_Original'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    steps_images['2_Grayscale'] = gray

    # 3. Blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    steps_images['3_Blurred'] = blurred

    # 4. Threshold (Tự động phát hiện nền)
    if np.mean(gray) > 128: 
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    else:
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    steps_images['4_Threshold'] = thresh

    # Contour
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, steps_images

    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    # 5. Box
    img_with_box = steps_images['1_Original'].copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
    steps_images['5_Contour_Box'] = img_with_box

    # 6. Crop & 7. Resize
    digit_crop = thresh[y:y+h, x:x+w]
    steps_images['6_Cropped'] = digit_crop
    
    width, height = digit_crop.shape[1], digit_crop.shape[0]
    if width > height:
        new_w = box_size; new_h = int(height * (new_w / width))
    else:
        new_h = box_size; new_w = int(width * (new_h / height))
    
    resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    steps_images['7_Resized_20px'] = resized

    # 8. Final
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    steps_images['8_Final_28x28'] = canvas

    final_pil_image = Image.fromarray(canvas)
    return final_pil_image, steps_images

def predict_top3(pil_image, model, device, class_names_map):
    """Dự đoán và trả về Top 3"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        idx = top3_idx[0][i].item()
        prob = top3_prob[0][i].item() * 100
        label = class_names_map.get(idx, f"Unknown {idx}")
        results.append({"label": label, "conf": prob})
    return results