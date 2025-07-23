import os
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision.models import swin_v2_s, Swin_V2_S_Weights

# GPU kontrolü
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Cihaz: {device}")


# CBAM Module Implementation
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = x * (avg_out + max_out)

        # Spatial attention
        spatial_avg = torch.mean(channel_out, dim=1, keepdim=True)
        spatial_max, _ = torch.max(channel_out, dim=1, keepdim=True)
        spatial_in = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_out = channel_out * self.spatial(spatial_in)

        return spatial_out


# BiFPN Module Implementation
class BiFPNBlock(nn.Module):
    def __init__(self, feature_sizes, num_channels):
        super().__init__()
        self.num_channels = num_channels

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(size, num_channels, 1)
            for size in feature_sizes
        ])

        # Top-down pathway
        self.top_down_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, 3, padding=1),
                nn.BatchNorm2d(num_channels),
                nn.ReLU()
            )
            for _ in range(len(feature_sizes) - 1)
        ])

        # Bottom-up pathway
        self.bottom_up_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, 3, padding=1),
                nn.BatchNorm2d(num_channels),
                nn.ReLU()
            )
            for _ in range(len(feature_sizes) - 1)
        ])

        # Output projections to ensure consistent channel size
        self.output_convs = nn.ModuleList([
            nn.Conv2d(num_channels, num_channels, 1)
            for _ in range(5)  # FPN uses 5 levels
        ])

        # Weights for feature fusion
        self.weights = nn.Parameter(torch.ones(len(feature_sizes), 2))
        self.eps = 1e-4

    def forward(self, features):
        # Handle dictionary input from ResNet backbone
        if isinstance(features, dict):
            # Extract features in order from ResNet (names are '0' through '4' or similar)
            feature_list = [features[str(i)] for i in sorted([int(k) for k in features.keys()])]
        else:
            feature_list = features

        # Convert input features to same channel size
        laterals = [conv(feature) for feature, conv in zip(feature_list, self.lateral_convs)]

        # Top-down pathway
        top_down = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            top = F.interpolate(top_down[-1], size=laterals[i].shape[2:], mode='nearest')
            w = torch.relu(self.weights[i])
            w = w / (torch.sum(w) + self.eps)
            top_down.append(self.top_down_convs[i](w[0] * laterals[i] + w[1] * top))

        top_down = top_down[::-1]

        # Bottom-up pathway
        bottom_up = [top_down[0]]
        for i in range(1, len(top_down)):
            w = torch.relu(self.weights[i])
            w = w / (torch.sum(w) + self.eps)
            bottom = F.max_pool2d(bottom_up[-1], kernel_size=2)
            bottom_up.append(self.bottom_up_convs[i - 1](w[0] * top_down[i] + w[1] * bottom))

        # Project to final feature maps
        # Ensure we have exactly 5 levels as expected by FPN
        while len(bottom_up) < 5:
            # Add additional levels through max pooling
            bottom_up.append(F.max_pool2d(bottom_up[-1], kernel_size=2, stride=2))

        # Take only the first 5 levels if we have more
        bottom_up = bottom_up[:5]

        # Apply output projections
        bottom_up = [conv(feature) for feature, conv in zip(bottom_up, self.output_convs)]

        # Convert to dictionary format for compatibility with FPN
        return_dict = {}
        for idx, feature in enumerate(bottom_up):
            return_dict[str(idx)] = feature

        return return_dict


# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return loss.mean()


# Veri setini yükle
def load_data_from_excel(excel_path):
    """Excel dosyasından veri yükler"""
    try:
        return pd.read_excel(excel_path)
    except Exception as e:
        print(f"Excel dosyası yüklenirken hata oluştu: {e}")
        return None

    # Özel Dataset sınıfı


class ShipDataset(Dataset):
    def __init__(self, df, image_ids, base_path="C:\\gemiv1\\data", transforms=None):
        self.df = df
        self.image_ids = image_ids
        self.base_path = base_path
        self.transforms = transforms

        # Sınıf isimlerini sayısal değerlere eşleştir
        self.class_map = {
            'Ship': 1,
            'Land': 2,
            'WS': 1  # WS de Ship olarak kabul edilir
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_data = self.df[self.df['idx_image'] == img_id]

        if img_data.empty:
            raise ValueError(f"Görüntü ID {img_id} için veri bulunamadı")

            # İlk satırdan görüntü yolunu al
        img_path = img_data.iloc[0]['path_image']

        # Yolu düzelt
        adjusted_path = img_path.replace('./Hüseyin Hoca/', self.base_path + '\\')
        adjusted_path = os.path.normpath(adjusted_path)

        if not os.path.exists(adjusted_path):
            raise FileNotFoundError(f"Görüntü bulunamadı: {adjusted_path}")

            # Görüntüyü oku
        img = cv2.imread(adjusted_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR'dan RGB'ye dönüştür

        # Dikdörtgenleri ve etiketleri hazırla
        boxes = []
        labels = []

        for _, row in img_data.iterrows():
            # Sınıf bilgisini al
            class_name = row['class_name']
            class_id = self.class_map.get(class_name, 1)  # Varsayılan olarak Ship (1)

            coords_str = row['str_dataPoint']
            coords = [int(x) for x in coords_str.split(',')]

            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append((coords[i], coords[i + 1]))

                    # Bounding box oluştur
            if len(points) >= 3:  # Poligon
                x_coords, y_coords = zip(*points)
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)

                # Geçerli bir kutu mu kontrol et
                if x_min < x_max and y_min < y_max:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)  # Her nesnenin kendi sınıf ID'sini kullan

            elif len(points) == 2:  # İki nokta (çizgi)
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                # Çizgi ince ise, biraz genişlet
                if x_min == x_max:
                    x_min = max(0, x_min - 2)
                    x_max = min(img.shape[1], x_max + 2)
                if y_min == y_max:
                    y_min = max(0, y_min - 2)
                    y_max = min(img.shape[0], y_max + 2)

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

            elif len(points) == 1:  # Tek nokta
                x, y = points[0]
                # 5x5 piksellik bir alan
                x_min = max(0, x - 2)
                y_min = max(0, y - 2)
                x_max = min(img.shape[1], x + 2)
                y_max = min(img.shape[0], y + 2)

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

                # Kutu ve etiketleri tensörlere dönüştür
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Eğer hiç kutu yoksa dummy bir kutu ekle (eğitimi bozmamak için)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

            # Target sözlüğü oluştur
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # Görüntüyü tensöre dönüştür
        img = TF.to_tensor(img)

        # Eğer transform varsa uygula
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # Model oluşturma fonksiyonu


def create_faster_rcnn_model(num_classes):
    # Load pre-trained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Add CBAM blocks to backbone
    backbone = model.backbone.body
    backbone.layer2.add_module('cbam2', CBAMBlock(512))
    backbone.layer3.add_module('cbam3', CBAMBlock(1024))
    backbone.layer4.add_module('cbam4', CBAMBlock(2048))

    # Replace FPN with BiFPN
    # ResNet50 feature sizes for each stage
    feature_sizes = [256, 512, 1024, 2048]  # C2, C3, C4, C5 from ResNet50
    bifpn = BiFPNBlock(feature_sizes, num_channels=256)  # 256 is the standard FPN channel size

    # Create a new backbone that combines the ResNet body with BiFPN
    class BackboneWithBiFPN(nn.Module):
        def __init__(self, body, bifpn):
            super().__init__()
            self.body = body
            self.bifpn = bifpn
            self.out_channels = 256  # Same as FPN output channels

        def forward(self, x):
            x = self.body(x)
            return self.bifpn(x)

    # Create new backbone
    new_backbone = BackboneWithBiFPN(backbone, bifpn)

    # Update the model's backbone
    model.backbone = new_backbone

    # Modify box predictor for num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Add Focal Loss
    model.roi_heads.box_predictor.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)

    return model


# Eğitim fonksiyonu
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_box_loss = 0

    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        # Classification loss (using Focal Loss)
        cls_losses = sum(loss for loss_name, loss in loss_dict.items() if 'cls' in loss_name)
        # Box regression loss
        box_losses = sum(loss for loss_name, loss in loss_dict.items() if 'box' in loss_name)

        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Update running loss values - ensure we're getting tensor values
        total_loss += losses.item()
        if isinstance(cls_losses, torch.Tensor):
            total_cls_loss += cls_losses.item()
        else:
            total_cls_loss += float(cls_losses)

        if isinstance(box_losses, torch.Tensor):
            total_box_loss += box_losses.item()
        else:
            total_box_loss += float(box_losses)

    # Calculate average losses
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_box_loss = total_box_loss / num_batches

    print(f"Epoch {epoch} - Total Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Box Loss: {avg_box_loss:.4f}")

    return avg_loss


# Değerlendirme fonksiyonu
def evaluate(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            # Burada istediğiniz metriği hesaplayabilirsiniz
            # Basitlik için şimdilik sadece tahminlerin yapıldığını kontrol edelim

    print("Değerlendirme tamamlandı")


# Görselleştirme fonksiyonu
def visualize_predictions(model, dataset, idx, device, score_threshold=0.5):
    model.eval()

    # Sınıf adları ve renkleri
    class_colors = {
        0: 'gray',  # Background
        1: 'red',  # Ship
        2: 'blue'  # Land
    }

    class_names = {
        0: 'Background',
        1: 'Ship',
        2: 'Land'
    }

    img, target = dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

        # Görüntüyü NumPy dizisine dönüştür
    img = img.permute(1, 2, 0).cpu().numpy()

    # Matplotlib figürü oluştur
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    # Ground truth kutuları
    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        color = class_colors.get(label.item(), 'green')
        class_name = class_names.get(label.item(), 'Unknown')

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"GT: {class_name}", color=color)

        # Tahmin edilen kutuları
    pred_boxes = prediction['boxes'].cpu().numpy()
    pred_scores = prediction['scores'].cpu().numpy()
    pred_labels = prediction['labels'].cpu().numpy()

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score >= score_threshold:
            x1, y1, x2, y2 = box
            color = class_colors.get(label.item(), 'orange')
            class_name = class_names.get(label.item(), 'Unknown')

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none',
                                     linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, y1 - 15, f"Pred: {class_name} {score:.2f}", color=color)

    plt.title(f"Görüntü {idx}: Ground Truth ve Tahminler")
    plt.show()


# ÖNEMLİ: DataLoader için collate_fn'i global scope'ta tanımlayın!
def collate_fn(batch):
    """DataLoader için batch oluşturma fonksiyonu"""
    return tuple(zip(*batch))


# Ana fonksiyon
def main():
    # Excel dosya yolu
    excel_path = r"C:\gemiv1\data\VeriDetay_v3.xlsx"

    # Veriyi yükle
    df = load_data_from_excel(excel_path)

    if df is None:
        print("Veri yüklenemedi, program sonlandırılıyor.")
        return

    print(f"Veri başarıyla yüklendi: {len(df)} satır.")

    # WS sınıfını Ship sınıfına dahil et
    df.loc[df['class_name'] == 'WS', 'class_name'] = 'Ship'

    print("\nBirleştirilmiş sınıf dağılımı:")
    print(df['class_name'].value_counts())

    # Benzersiz görüntü ID'lerini al
    unique_image_ids = df['idx_image'].unique()

    # Veriyi train ve validation olarak böl
    train_ids, val_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)

    print(f"Toplam {len(unique_image_ids)} görüntü: {len(train_ids)} eğitim, {len(val_ids)} doğrulama")

    # Dataset oluştur
    train_dataset = ShipDataset(df, train_ids)
    val_dataset = ShipDataset(df, val_ids)

    # DataLoader oluştur - batch size'ı küçült (CBAM ve BiFPN nedeniyle)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Reduced from 16 due to increased memory usage
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

    # Model oluştur (3 sınıf: background, ship ve land)
    model = create_faster_rcnn_model(num_classes=3)
    model.to(device)

    # Optimizer - AdamW kullan ve learning rate'i düşür
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0001)

    # Cosine Annealing scheduler kullan
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=15,  # 15 epoch'ta minimum değere ulaş
        eta_min=1e-6  # Minimum learning rate
    )

    # Eğitim döngüsü - epoch sayısını artır
    num_epochs = 30  # Increased from 10
    best_loss = float('inf')
    patience = 3  # Early stopping patience
    patience_counter = 0

    for epoch in range(num_epochs):
        # Eğitim
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Learning rate güncelleme
        scheduler.step()

        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # En iyi modeli kaydet
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, "faster_rcnn_gemi_best.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch}")
            break

        # Her 5 epoch'ta bir modeli kaydet
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"faster_rcnn_gemi_epoch_{epoch + 1}.pth")

    # Son modeli kaydet
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, "faster_rcnn_gemi_son.pth")

    print("Eğitim tamamlandı, model kaydedildi.")

    # Değerlendirme
    print("Model değerlendiriliyor...")
    evaluate(model, val_loader, device)

    # Bazı tahminleri görselleştir
    num_samples = min(5, len(val_dataset))
    for i in range(num_samples):
        idx = np.random.randint(0, len(val_dataset))
        print(f"Görüntü {idx} için tahminler gösteriliyor...")
        visualize_predictions(model, val_dataset, idx, device)


if __name__ == "__main__":
    main()