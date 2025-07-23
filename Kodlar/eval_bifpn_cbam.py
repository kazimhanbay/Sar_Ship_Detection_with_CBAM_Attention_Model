import os
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F  # For image transforms
import torch.nn.functional as F_nn  # For network operations like interpolate
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import torch.nn as nn

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


# BiFPN Module Implementation - FIXED
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

        # Output projections for consistent channels
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
            # Extract features in order from ResNet
            feature_list = [features[str(i)] for i in sorted([int(k) for k in features.keys()])]
        else:
            feature_list = features

        # Convert input features to same channel size
        laterals = [conv(feature) for feature, conv in zip(feature_list, self.lateral_convs)]

        # Top-down pathway - FIXED: Use F_nn.interpolate
        top_down = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            # Use F_nn instead of F for interpolate
            top = F_nn.interpolate(top_down[-1], size=laterals[i].shape[2:], mode='nearest')
            w = torch.relu(self.weights[i])
            w = w / (torch.sum(w) + self.eps)
            top_down.append(self.top_down_convs[i](w[0] * laterals[i] + w[1] * top))

        top_down = top_down[::-1]

        # Bottom-up pathway
        bottom_up = [top_down[0]]
        for i in range(1, len(top_down)):
            w = torch.relu(self.weights[i])
            w = w / (torch.sum(w) + self.eps)
            bottom = F_nn.max_pool2d(bottom_up[-1], kernel_size=2)
            bottom_up.append(self.bottom_up_convs[i - 1](w[0] * top_down[i] + w[1] * bottom))

        # Ensure we have exactly 5 levels
        while len(bottom_up) < 5:
            bottom_up.append(F_nn.max_pool2d(bottom_up[-1], kernel_size=2, stride=2))

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
        ce_loss = F_nn.cross_entropy(inputs, targets, reduction='none')
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


# IoU (Intersection over Union) hesaplama
def box_iou(box1, box2):
    """İki bounding box arasındaki IoU'yu hesaplar"""
    # Box koordinatları: [x1, y1, x2, y2]

    # Kesişim koordinatlarını hesapla
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Kesişim alanı
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Box alanları
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Birleşim alanı
    union = box1_area + box2_area - intersection

    # IoU hesapla
    iou = intersection / union if union > 0 else 0

    return iou


# Merkez noktası mesafe hesaplama
def calculate_center_distance(box1, box2):
    """İki kutu arasındaki merkez noktaları arasındaki mesafeyi hesaplar"""
    # Kutu merkezlerini hesapla
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2

    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2

    # Öklid mesafesi
    distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
    return distance


# MAP hesaplama yardımcı fonksiyonları
def calculate_ap(precision, recall):
    """Precision-Recall eğrisi altındaki alanı (Average Precision) hesaplar"""
    # 11-point interpolation AP hesabı
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11
    return ap


# Özel Dataset sınıfı
class ShipDataset(Dataset):
    def __init__(self, df, image_ids, base_path="C:\\gemiv1\\data", transforms=None, box_expansion=10):
        self.df = df
        self.image_ids = image_ids
        self.base_path = base_path
        self.transforms = transforms
        self.box_expansion = box_expansion  # Kutu genişletme miktarı (piksel)

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
        img_height, img_width = img.shape[:2]

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

            # Bounding box oluştur ve genişlet
            if len(points) >= 3:  # Poligon
                x_coords, y_coords = zip(*points)
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)

                # Kutuyu genişlet
                x_min = max(0, x_min - self.box_expansion)
                y_min = max(0, y_min - self.box_expansion)
                x_max = min(img_width, x_max + self.box_expansion)
                y_max = min(img_height, y_max + self.box_expansion)

                # Geçerli bir kutu mu kontrol et
                if x_min < x_max and y_min < y_max:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

            elif len(points) == 2:  # İki nokta (çizgi)
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                # Çizgi ince ise, daha fazla genişlet
                if x_min == x_max:
                    x_min = max(0, x_min - self.box_expansion)
                    x_max = min(img_width, x_max + self.box_expansion)
                if y_min == y_max:
                    y_min = max(0, y_min - self.box_expansion)
                    y_max = min(img_height, y_max + self.box_expansion)

                # Her durumda biraz daha genişlet
                x_min = max(0, x_min - self.box_expansion)
                y_min = max(0, y_min - self.box_expansion)
                x_max = min(img_width, x_max + self.box_expansion)
                y_max = min(img_height, y_max + self.box_expansion)

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)

            elif len(points) == 1:  # Tek nokta
                x, y = points[0]
                # Daha büyük bir alan
                expansion = max(5, self.box_expansion * 2)  # Tek noktalar için daha büyük genişletme
                x_min = max(0, x - expansion)
                y_min = max(0, y - expansion)
                x_max = min(img_width, x + expansion)
                y_max = min(img_height, y + expansion)

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
        target["image_id"] = torch.tensor([img_id])  # Orijinal image_id kullan
        target["image_path"] = adjusted_path  # Görüntü yolunu da ekleyelim

        # Görüntüyü tensöre dönüştür
        img = F.to_tensor(img)

        # Eğer transform varsa uygula
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# Model oluşturma fonksiyonu - CBAM ve BiFPN modüllerini ekleyen versiyonu oluştur
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
    bifpn = BiFPNBlock(feature_sizes, num_channels=256)  # 256 is standard FPN channel size

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

    return model


# DataLoader için collate_fn fonksiyonu
def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_model(model, data_loader, device, output_dir="C:\\gemiv1\\sonuclar_gemi_tespiti_cbam_bifpn",
                   conf_threshold=0.5, iou_threshold=0.3, distance_threshold=50, use_distance_metric=True):
    """
    Eğitilmiş modeli değerlendirir, sonuçları görselleştirir ve metrikleri Excel'e kaydeder
    """
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Sınıf adları ve renkleri
    class_names = {
        0: 'Background',
        1: 'Ship',
        2: 'Land'
    }

    class_colors = {
        0: 'gray',
        1: 'red',
        2: 'blue'
    }

    # Performans metrikleri
    total_time = 0
    num_processed = 0

    # Sınıf bazlı metrikler
    class_metrics = {cls_id: {'total_gt': 0, 'total_pred': 0, 'true_positives': 0,
                              'false_positives': 0, 'false_negatives': 0}
                     for cls_id in [1, 2]}  # Ship ve Land için

    # Görüntü başına metrikleri saklamak için DataFrame
    image_metrics = []

    # mAP hesabı için sınıf bazlı listeler
    class_predictions = {cls_id: [] for cls_id in [1, 2]}
    class_scores = {cls_id: [] for cls_id in [1, 2]}

    # Gemiler için özel metrikler - GÜNCELLENDİ
    ship_detection_metrics = {
        'total_ships': 0,  # Görüntülerdeki toplam gemi sayısı (ground truth)
        'correctly_detected_ships': 0,  # Doğru tespit edilen bireysel gemi sayısı (YENİ)
        'total_detected': 0,  # Tespit edilen toplam gemi sayısı
        'ship_detection_rate': 0,  # Bireysel gemi tespit oranı (YENİ)
        'images_with_ships': 0,  # Gemi içeren görüntü sayısı
        'correct_ship_images': 0,  # En az bir gemiyi doğru tespit ettiğimiz görüntü sayısı
        'image_detection_rate': 0  # Görüntü bazlı tespit oranı (eski detection_rate)
    }

    model.eval()

    print(f"Toplam {len(data_loader)} batch değerlendirilecek...")
    print(f"Kullanılan metrik: {'Merkez Mesafesi' if use_distance_metric else 'IoU'}")
    print(f"Mesafe eşiği: {distance_threshold} piksel" if use_distance_metric else f"IoU eşiği: {iou_threshold}")

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Görüntüler değerlendiriliyor"):
            # Görüntüleri GPU'ya taşı
            images = list(img.to(device) for img in images)

            # Tahmin zamanını ölç
            start_time = time.time()

            # Tahminleri yap
            outputs = model(images)

            # Tahmin süresini kaydet
            inference_time = time.time() - start_time
            total_time += inference_time

            # Her bir görüntü için tahminleri işle
            for i, (output, target) in enumerate(zip(outputs, targets)):
                num_processed += 1
                image_id = target["image_id"].item()
                image_path = target["image_path"]

                # Orijinal görüntüyü yükle (görsellleştirme için)
                original_img = cv2.imread(image_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                # Ground truth kutuları ve etiketleri
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                # Gemi metrikleri için bayrağa ihtiyacımız var
                image_has_ships = np.any(gt_labels == 1)
                image_detected_ship = False

                if image_has_ships:
                    ship_detection_metrics['images_with_ships'] += 1
                    # Görüntüdeki gemi sayısını güncelle
                    ship_count = np.sum(gt_labels == 1)
                    ship_detection_metrics['total_ships'] += ship_count

                # Her bir sınıf için ground truth sayısını güncelle
                for label in gt_labels:
                    if label in class_metrics:
                        class_metrics[label]['total_gt'] += 1

                # Tahmin kutularını ve etiketlerini al
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                # Eşik değerinden yüksek tahminleri filtrele
                keep = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                # Her bir sınıf için tahmin sayısını güncelle
                for label in pred_labels:
                    if label in class_metrics:
                        class_metrics[label]['total_pred'] += 1

                if len(pred_labels[pred_labels == 1]) > 0:
                    ship_detection_metrics['total_detected'] += len(pred_labels[pred_labels == 1])

                # Sınıf bazlı eşleşme hesapla
                # Her bir sınıf için ayrı ayrı eşleşme bul
                for cls_id in [1, 2]:  # Ship ve Land
                    # Bu sınıfa ait ground truth ve tahminleri filtrele
                    cls_gt_indices = np.where(gt_labels == cls_id)[0]
                    cls_pred_indices = np.where(pred_labels == cls_id)[0]

                    cls_gt_boxes = gt_boxes[cls_gt_indices]
                    cls_pred_boxes = pred_boxes[cls_pred_indices]
                    cls_pred_scores = pred_scores[cls_pred_indices]

                    # Bu görüntüdeki doğru tespit edilmiş gemi sayısını sıfırla
                    img_true_positives = 0

                    # İleri metrik hesaplamalarını yap
                    if use_distance_metric and cls_id == 1:  # Gemiler için merkez mesafesi kullan
                        # Merkez mesafesi eşleştirme matrisi
                        distance_matrix = np.zeros((len(cls_gt_boxes), len(cls_pred_boxes)))
                        for gt_idx, gt_box in enumerate(cls_gt_boxes):
                            for pred_idx, pred_box in enumerate(cls_pred_boxes):
                                distance_matrix[gt_idx, pred_idx] = calculate_center_distance(gt_box, pred_box)

                        # Eşleşmeleri bul (mesafe < distance_threshold)
                        matched_indices = []
                        for gt_idx in range(len(cls_gt_boxes)):
                            best_match_idx = -1
                            best_match_distance = distance_threshold

                            for pred_idx in range(len(cls_pred_boxes)):
                                # Eğer bu tahmin başka bir ground truth ile eşleşmişse atla
                                if any(pred_idx == match[1] for match in matched_indices):
                                    continue

                                distance = distance_matrix[gt_idx, pred_idx]
                                if distance < best_match_distance:
                                    best_match_distance = distance
                                    best_match_idx = pred_idx

                            if best_match_idx >= 0:
                                matched_indices.append((gt_idx, best_match_idx))
                                image_detected_ship = True
                                img_true_positives += 1
                    else:
                        # Standart IoU eşleştirme
                        iou_matrix = np.zeros((len(cls_gt_boxes), len(cls_pred_boxes)))
                        for gt_idx, gt_box in enumerate(cls_gt_boxes):
                            for pred_idx, pred_box in enumerate(cls_pred_boxes):
                                iou_matrix[gt_idx, pred_idx] = box_iou(gt_box, pred_box)

                        # Eşleşmeleri bul (IoU > threshold)
                        matched_indices = []
                        for gt_idx in range(len(cls_gt_boxes)):
                            # En iyi eşleşmeyi bul
                            best_match_idx = -1
                            best_match_iou = iou_threshold

                            for pred_idx in range(len(cls_pred_boxes)):
                                # Eğer bu tahmin başka bir ground truth ile eşleşmişse atla
                                if any(pred_idx == match[1] for match in matched_indices):
                                    continue

                                iou = iou_matrix[gt_idx, pred_idx]
                                if iou >= best_match_iou:
                                    best_match_iou = iou
                                    best_match_idx = pred_idx

                            if best_match_idx >= 0:
                                matched_indices.append((gt_idx, best_match_idx))
                                if cls_id == 1:  # Ship
                                    image_detected_ship = True
                                    img_true_positives += 1

                    # Metrikleri hesapla
                    true_positives = len(matched_indices)
                    false_positives = len(cls_pred_boxes) - true_positives
                    false_negatives = len(cls_gt_boxes) - true_positives

                    # Sınıf metriklerini güncelle
                    class_metrics[cls_id]['true_positives'] += true_positives
                    class_metrics[cls_id]['false_positives'] += false_positives
                    class_metrics[cls_id]['false_negatives'] += false_negatives

                    # GEMİ SINIFI İÇİN: Bireysel gemi tespit sayısını güncelle
                    if cls_id == 1:  # Ship
                        ship_detection_metrics['correctly_detected_ships'] += true_positives

                    # mAP için verileri sakla
                    for pred_idx in range(len(cls_pred_boxes)):
                        is_true_positive = any(match[1] == pred_idx for match in matched_indices)
                        class_predictions[cls_id].append(1 if is_true_positive else 0)
                        class_scores[cls_id].append(cls_pred_scores[pred_idx])

                # Görüntü gemi içeriyorsa ve en az bir gemi tespit edilmişse sayacı artır
                if image_has_ships and image_detected_ship:
                    ship_detection_metrics['correct_ship_images'] += 1

                # Görüntü başına genel metrikleri hesapla
                total_true_positives = sum(class_metrics[cls_id]['true_positives'] for cls_id in [1, 2])
                total_false_positives = sum(class_metrics[cls_id]['false_positives'] for cls_id in [1, 2])
                total_false_negatives = sum(class_metrics[cls_id]['false_negatives'] for cls_id in [1, 2])

                total_precision = total_true_positives / (total_true_positives + total_false_positives) if (
                                                                                                                   total_true_positives + total_false_positives) > 0 else 0
                total_recall = total_true_positives / (total_true_positives + total_false_negatives) if (
                                                                                                                total_true_positives + total_false_negatives) > 0 else 0
                total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (
                                                                                                            total_precision + total_recall) > 0 else 0

                # Gemi bulma başarısı
                ship_found_percent = 100 if image_has_ships and image_detected_ship else 0

                # Görüntü başına metrikleri kaydet
                image_metrics.append({
                    'image_id': image_id,
                    'ship_gt_count': sum(1 for label in gt_labels if label == 1),
                    'land_gt_count': sum(1 for label in gt_labels if label == 2),
                    'ship_pred_count': sum(1 for label in pred_labels if label == 1),
                    'land_pred_count': sum(1 for label in pred_labels if label == 2),
                    'ship_true_positives': class_metrics[1]['true_positives'],
                    'land_true_positives': class_metrics[2]['true_positives'],
                    'ship_false_positives': class_metrics[1]['false_positives'],
                    'land_false_positives': class_metrics[2]['false_positives'],
                    'ship_false_negatives': class_metrics[1]['false_negatives'],
                    'land_false_negatives': class_metrics[2]['false_negatives'],
                    'total_precision': total_precision,
                    'total_recall': total_recall,
                    'total_f1': total_f1,
                    'ship_found': ship_found_percent,  # Gemi bulma başarısı
                    'inference_time_ms': inference_time * 1000 / len(images)
                })

                # Görselleştirme için figür oluştur (2 alt figür: Ground Truth ve Tahminler)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

                # Sol figür: Ground Truth
                ax1.imshow(original_img)
                ax1.set_title("Ground Truth")
                # Ground truth kutuları (yeşil=Ship, mavi=Land)
                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    color = class_colors.get(label, 'green')
                    class_name = class_names.get(label, 'Unknown')

                    rect = patches.Rectangle((x1, y1), width, height,
                                             linewidth=2, edgecolor=color,
                                             facecolor='none')
                    ax1.add_patch(rect)
                    ax1.text(x1, y1 - 5, f"{class_name}", color=color, fontsize=8)

                    # Merkez noktasını göster
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    ax1.plot(center_x, center_y, 'o', color=color, markersize=5)

                ax1.axis('off')

                # Sağ figür: Tahminler
                ax2.imshow(original_img)
                title_suffix = "Merkez Mesafesi" if use_distance_metric else "IoU"
                ax2.set_title(f"Model Tahminleri ({title_suffix})")

                # Tahmin kutuları (kırmızı=Ship, mavi=Land)
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    color = class_colors.get(label, 'orange')
                    class_name = class_names.get(label, 'Unknown')

                    rect = patches.Rectangle((x1, y1), width, height,
                                             linewidth=2, edgecolor=color,
                                             facecolor='none')
                    ax2.add_patch(rect)
                    ax2.text(x1, y1 - 10, f"{class_name}: {score:.2f}", color=color, fontsize=8)

                    # Merkez noktasını göster
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    ax2.plot(center_x, center_y, 'o', color=color, markersize=5)

                ax2.axis('off')

                # Merkez mesafesi için çizgileri göster (sadece gemiler için)
                if use_distance_metric and image_has_ships:
                    ship_gt_indices = np.where(gt_labels == 1)[0]
                    ship_pred_indices = np.where(pred_labels == 1)[0]

                    ship_gt_boxes = gt_boxes[ship_gt_indices]
                    ship_pred_boxes = pred_boxes[ship_pred_indices]

                    # İyi eşleşmeleri çizgiyle göster
                    distance_matrix = np.zeros((len(ship_gt_boxes), len(ship_pred_boxes)))
                    for gt_idx, gt_box in enumerate(ship_gt_boxes):
                        for pred_idx, pred_box in enumerate(ship_pred_boxes):
                            distance = calculate_center_distance(gt_box, pred_box)
                            distance_matrix[gt_idx, pred_idx] = distance

                            # Eğer mesafe eşikten küçükse, merkez noktalarını bir çizgiyle bağla
                            if distance < distance_threshold:
                                gt_center_x = (gt_box[0] + gt_box[2]) / 2
                                gt_center_y = (gt_box[1] + gt_box[3]) / 2
                                pred_center_x = (pred_box[0] + pred_box[2]) / 2
                                pred_center_y = (pred_box[1] + pred_box[3]) / 2

                                ax2.plot([gt_center_x, pred_center_x], [gt_center_y, pred_center_y],
                                         color='green', linestyle='--', linewidth=1, alpha=0.7)
                                ax2.text((gt_center_x + pred_center_x) / 2, (gt_center_y + pred_center_y) / 2,
                                         f"{distance:.1f}px", fontsize=7, color='green')

                # Eğer gemiler bulunduysa yeşil, bulunamadıysa kırmızı çerçeve
                ship_detection_status = "Bulundu ✓" if image_has_ships and image_detected_ship else "Bulunamadı ✗"
                status_color = 'green' if image_has_ships and image_detected_ship else 'red'

                # Genel başlık ekle
                plt.suptitle(
                    f"Görüntü ID: {image_id} - Gemi: {ship_detection_status}",
                    fontsize=14, color=status_color)

                # Figürü kaydet
                output_path = os.path.join(output_dir, f"sonuc_id{image_id}.png")
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

                # OpenCV ile de kaydet (daha hızlı göz atabilmek için)
                cv_img_gt = original_img.copy()
                cv_img_pred = original_img.copy()

                # Ground truth kutuları
                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = map(int, box)
                    if label == 1:  # Ship
                        cv2.rectangle(cv_img_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Kırmızı
                        cv2.putText(cv_img_gt, "Ship", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        # Merkez noktasını çiz
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        cv2.circle(cv_img_gt, (center_x, center_y), 3, (255, 0, 0), -1)
                    elif label == 2:  # Land
                        cv2.rectangle(cv_img_gt, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Mavi
                        cv2.putText(cv_img_gt, "Land", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Tahmin kutuları
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    x1, y1, x2, y2 = map(int, box)
                    if label == 1:  # Ship
                        cv2.rectangle(cv_img_pred, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Kırmızı
                        cv2.putText(cv_img_pred, f"Ship: {score:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        # Merkez noktasını çiz
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        cv2.circle(cv_img_pred, (center_x, center_y), 3, (255, 0, 0), -1)
                    elif label == 2:  # Land
                        cv2.rectangle(cv_img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Mavi
                        cv2.putText(cv_img_pred, f"Land: {score:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # İki görüntüyü yan yana koy
                cv_combined = np.hstack((cv_img_gt, cv_img_pred))

                # OpenCV görüntüsünü kaydet
                cv_output_path = os.path.join(output_dir, f"cv_sonuc_id{image_id}.jpg")
                cv_combined_bgr = cv2.cvtColor(cv_combined, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cv_output_path, cv_combined_bgr)

    # GÜNCELLENDİ: Her iki tespit oranını da hesapla
    # 1. Bireysel gemilerin tespit oranı
    if ship_detection_metrics['total_ships'] > 0:
        ship_detection_metrics['ship_detection_rate'] = ship_detection_metrics['correctly_detected_ships'] / \
                                                        ship_detection_metrics['total_ships']

    # 2. Görüntü bazlı tespit oranı (eski detection_rate)
    if ship_detection_metrics['images_with_ships'] > 0:
        ship_detection_metrics['image_detection_rate'] = ship_detection_metrics['correct_ship_images'] / \
                                                         ship_detection_metrics['images_with_ships']

    # Sınıf bazlı mAP hesapla
    class_aps = {}

    for cls_id in [1, 2]:  # Ship ve Land
        if len(class_scores[cls_id]) == 0:
            class_aps[cls_id] = 0.0
            continue

        # Tahminleri skorlarına göre sırala
        sorted_indices = np.argsort(class_scores[cls_id])[::-1]  # Büyükten küçüğe sırala
        class_predictions[cls_id] = np.array(class_predictions[cls_id])[sorted_indices]

        # Precision ve Recall hesapla
        cumulative_true_positives = np.cumsum(class_predictions[cls_id])
        cumulative_false_positives = np.cumsum(1 - class_predictions[cls_id])

        precision_curve = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
        recall_curve = cumulative_true_positives / class_metrics[cls_id]['total_gt'] if class_metrics[cls_id][
                                                                                            'total_gt'] > 0 else np.zeros_like(
            cumulative_true_positives)

        # AP (Average Precision) hesapla
        class_aps[cls_id] = calculate_ap(precision_curve, recall_curve)

    # Genel mAP (mean Average Precision) hesapla
    mAP = np.mean([ap for ap in class_aps.values()])

    # Sınıf bazlı metrikleri hesapla
    class_performance = []

    for cls_id in [1, 2]:  # Ship ve Land
        tp = class_metrics[cls_id]['true_positives']
        fp = class_metrics[cls_id]['false_positives']
        fn = class_metrics[cls_id]['false_negatives']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_performance.append({
            'class_id': cls_id,
            'class_name': class_names[cls_id],
            'total_gt': class_metrics[cls_id]['total_gt'],
            'total_pred': class_metrics[cls_id]['total_pred'],
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ap': class_aps[cls_id]
        })

    # Performans istatistiklerini yazdır
    if num_processed > 0:
        avg_time = total_time / num_processed
        fps = 1.0 / avg_time

        print(f"\n--- Performans İstatistikleri ---")
        print(f"Toplam işlenen görüntü: {num_processed}")
        print(f"mAP@{iou_threshold}: {mAP:.4f}")
        print(f"Ortalama işlem süresi: {avg_time * 1000:.1f} ms/görüntü")
        print(f"FPS: {fps:.2f}")

        print("\n--- Sınıf Bazlı Performans ---")
        for cls_perf in class_performance:
            cls_name = cls_perf['class_name']
            print(f"\nSınıf: {cls_name}")
            print(f"  Ground Truth: {cls_perf['total_gt']}")
            print(f"  Tahmin: {cls_perf['total_pred']}")
            print(f"  Precision: {cls_perf['precision']:.4f}")
            print(f"  Recall: {cls_perf['recall']:.4f}")
            print(f"  F1-Score: {cls_perf['f1_score']:.4f}")
            print(f"  AP: {cls_perf['ap']:.4f}")

        # Gemi tespit metrikleri - GÜNCELLENDİ
        print("\n--- Gemi Tespit İstatistikleri ---")
        print(f"Toplam gemi: {ship_detection_metrics['total_ships']}")
        print(f"Doğru tespit edilen gemi sayısı: {ship_detection_metrics['correctly_detected_ships']}")
        print(f"BİREYSEL GEMİ BAZLI TESPİT ORANI: {ship_detection_metrics['ship_detection_rate'] * 100:.2f}%")
        print(f"Gemi içeren görüntü: {ship_detection_metrics['images_with_ships']}")
        print(f"En az bir gemi doğru tespit edilen görüntü: {ship_detection_metrics['correct_ship_images']}")
        print(f"GÖRÜNTÜ BAZLI TESPİT ORANI: {ship_detection_metrics['image_detection_rate'] * 100:.2f}%")

        # Excel'e kaydet
        # 1. Genel performans özeti
        performance_summary = {
            'total_images': num_processed,
            'mAP': mAP,
            'ship_AP': class_aps[1],
            'land_AP': class_aps[2],
            'average_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'distance_threshold': distance_threshold if use_distance_metric else "N/A",
            'use_distance_metric': use_distance_metric
        }

        performance_df = pd.DataFrame([performance_summary])

        # 2. Sınıf bazlı performans
        class_performance_df = pd.DataFrame(class_performance)

        # 3. Görüntü başına metrikler
        image_metrics_df = pd.DataFrame(image_metrics)

        # 4. Gemi tespit metrikleri - GÜNCELLENDİ
        ship_metrics_df = pd.DataFrame([ship_detection_metrics])

        # Excel'e kaydet
        with pd.ExcelWriter(os.path.join(output_dir, 'metrik_sonuclari.xlsx')) as writer:
            performance_df.to_excel(writer, sheet_name='Genel Performans', index=False)
            class_performance_df.to_excel(writer, sheet_name='Sınıf Performansı', index=False)
            image_metrics_df.to_excel(writer, sheet_name='Görüntü Metrikleri', index=False)
            ship_metrics_df.to_excel(writer, sheet_name='Gemi Tespit Oranı', index=False)

        print(f"Metrikler Excel dosyasına kaydedildi: {os.path.join(output_dir, 'metrik_sonuclari.xlsx')}")


# DataLoader için collate_fn fonksiyonu
def collate_fn(batch):
    return tuple(zip(*batch))


# Ana fonksiyon
def main():
    # Excel dosya yolu
    excel_path = r"C:\gemiv1\data\VeriDetay_v3.xlsx"

    # Çıktı klasörü
    output_dir = r"C:\gemiv1\sonuclar_gemi_tespiti_bipfn_cbam"

    # Veriyi yükle
    df = load_data_from_excel(excel_path)

    if df is None:
        print("Veri yüklenemedi, program sonlandırılıyor.")
        return

    print(f"Veri başarıyla yüklendi: {len(df)} satır.")

    # Sınıf dağılımını kontrol et
    print("\nSınıf dağılımı:")
    print(df['class_name'].value_counts())

    # Benzersiz görüntü ID'lerini al
    unique_image_ids = df['idx_image'].unique()

    # Veriyi train ve validation olarak böl
    train_ids, val_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)

    print(f"Toplam {len(unique_image_ids)} görüntü: {len(train_ids)} eğitim, {len(val_ids)} doğrulama")

    # Burada sadece validation veri setini değerlendireceğiz - bounding box'ları genişletilmiş
    val_dataset = ShipDataset(df, val_ids, box_expansion=15)

    # Değerlendirme DataLoader'ı oluştur
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # Değerlendirme için daha küçük batch size tercih edilir
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Model oluştur (3 sınıf: background, ship ve land)
    model = create_faster_rcnn_model(num_classes=3)

    # Eğitilmiş modeli yükle - ÖNEMLİ: Tam checkpoint'ten yüklüyoruz
    model_path = "faster_rcnn_gemi_son.pth"
    checkpoint = torch.load(model_path)

    # Model state_dict'i kontrol et ve yükle
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model yüklendi: checkpoint['model_state_dict']")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Model yüklendi: checkpoint['state_dict']")
    else:
        try:
            # Belki doğrudan model state dict'tir
            model.load_state_dict(checkpoint)
            print("Model doğrudan state_dict olarak yüklendi")
        except:
            print("UYARI: Model yüklenemedi! Checkpoint formatı beklenen yapıda değil.")
            print("Checkpoint anahtarları:", checkpoint.keys())
            # Basit bir modelle devam et
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

    model.to(device)

    # Modeli değerlendir
    print("Model değerlendiriliyor...")
    evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        output_dir=output_dir,
        conf_threshold=0.5,  # Nesne algılama için güven eşiği
        iou_threshold=0.3,  # TP/FP için IoU eşiği (mesafe metriği kullanılırken dikkate alınmaz)
        distance_threshold=50,  # Merkez noktaları arası maksimum mesafe (piksel)
        use_distance_metric=True  # IoU yerine merkez mesafesi kullan
    )

    print(f"Sonuçlar {output_dir} klasörüne kaydedildi.")


if __name__ == "__main__":
    main()