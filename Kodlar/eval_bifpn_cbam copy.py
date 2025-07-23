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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import time
import torch.nn as nn
import datetime
import seaborn as sns

# GPU kontrolü
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")


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
    Evaluates the trained model, visualizes results, and saves metrics to Excel
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Class names in English
    class_names = {
        0: 'Background',
        1: 'Ship',
        2: 'Land'
    }

    # Updated colors: green for ground truth, red for predictions
    gt_color = 'green'
    pred_color = 'red'
    land_color = 'blue'  # Keep land as blue for better distinction

    # Performance metrics
    total_time = 0
    num_processed = 0

    # Class-based metrics
    class_metrics = {cls_id: {'total_gt': 0, 'total_pred': 0, 'true_positives': 0,
                              'false_positives': 0, 'false_negatives': 0}
                     for cls_id in [1, 2]}  # For Ship and Land

    # DataFrame to store per-image metrics
    image_metrics = []

    # Lists for class-based mAP calculation
    class_predictions = {cls_id: [] for cls_id in [1, 2]}
    class_scores = {cls_id: [] for cls_id in [1, 2]}

    # For confusion matrix
    all_gt_labels = []
    all_pred_labels = []

    # Special metrics for ships - UPDATED
    ship_detection_metrics = {
        'total_ships': 0,  # Total number of ships in images (ground truth)
        'correctly_detected_ships': 0,  # Correctly detected individual ships (NEW)
        'total_detected': 0,  # Total number of detected ships
        'ship_detection_rate': 0,  # Individual ship detection rate (NEW)
        'images_with_ships': 0,  # Number of images containing ships
        'correct_ship_images': 0,  # Number of images where we correctly detected at least one ship
        'image_detection_rate': 0  # Image-based detection rate (old detection_rate)
    }

    model.eval()

    print(f"Evaluating {len(data_loader)} batches...")
    print(f"Metric used: {'Center Distance' if use_distance_metric else 'IoU'}")
    print(f"Distance threshold: {distance_threshold} pixels" if use_distance_metric else f"IoU threshold: {iou_threshold}")

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

                # Get prediction boxes and labels
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                # Filter predictions above threshold
                keep = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                # For each class, keep track of total predictions
                for label in pred_labels:
                    if label in class_metrics:
                        class_metrics[label]['total_pred'] += 1

                if len(pred_labels[pred_labels == 1]) > 0:
                    ship_detection_metrics['total_detected'] += len(pred_labels[pred_labels == 1])

                # Find class-based matches
                # Calculate matches for each class separately
                matched_gt_indices = {}  # Track which ground truths are matched
                
                # Store the best detection for each ground truth (for visualization)
                best_detections = {}
                
                for cls_id in [1, 2]:  # Ship and Land
                    # Filter ground truth and predictions for this class
                    cls_gt_indices = np.where(gt_labels == cls_id)[0]
                    cls_pred_indices = np.where(pred_labels == cls_id)[0]

                    cls_gt_boxes = gt_boxes[cls_gt_indices]
                    cls_pred_boxes = pred_boxes[cls_pred_indices]
                    cls_pred_scores = pred_scores[cls_pred_indices]

                    # Reset true positives counter for this image
                    img_true_positives = 0

                    # Advanced metric calculations
                    if use_distance_metric and cls_id == 1:  # Use center distance for ships
                        # Center distance matching matrix
                        distance_matrix = np.zeros((len(cls_gt_boxes), len(cls_pred_boxes)))
                        for gt_idx, gt_box in enumerate(cls_gt_boxes):
                            for pred_idx, pred_box in enumerate(cls_pred_boxes):
                                distance_matrix[gt_idx, pred_idx] = calculate_center_distance(gt_box, pred_box)

                        # Find matches (distance < distance_threshold)
                        matched_indices = []
                        matched_gt_indices[cls_id] = set()
                        
                        for gt_idx in range(len(cls_gt_boxes)):
                            best_match_idx = -1
                            best_match_distance = distance_threshold
                            best_match_score = -1

                            for pred_idx in range(len(cls_pred_boxes)):
                                # Skip if prediction is already matched to another ground truth
                                if any(pred_idx == match[1] for match in matched_indices):
                                    continue

                                distance = distance_matrix[gt_idx, pred_idx]
                                score = cls_pred_scores[pred_idx]
                                
                                if distance < best_match_distance:
                                    best_match_distance = distance
                                    best_match_idx = pred_idx
                                    best_match_score = score

                            if best_match_idx >= 0:
                                matched_indices.append((gt_idx, best_match_idx))
                                matched_gt_indices[cls_id].add(cls_gt_indices[gt_idx])
                                image_detected_ship = True
                                img_true_positives += 1
                                
                                # Store best detection for this ground truth
                                gt_global_idx = cls_gt_indices[gt_idx]
                                pred_global_idx = cls_pred_indices[best_match_idx]
                                best_detections[(gt_global_idx, cls_id)] = (pred_global_idx, best_match_score)
                    else:
                        # Standard IoU matching
                        iou_matrix = np.zeros((len(cls_gt_boxes), len(cls_pred_boxes)))
                        for gt_idx, gt_box in enumerate(cls_gt_boxes):
                            for pred_idx, pred_box in enumerate(cls_pred_boxes):
                                iou_matrix[gt_idx, pred_idx] = box_iou(gt_box, pred_box)

                        # Find matches (IoU > threshold)
                        matched_indices = []
                        matched_gt_indices[cls_id] = set()
                        
                        for gt_idx in range(len(cls_gt_boxes)):
                            # Find best match
                            best_match_idx = -1
                            best_match_iou = iou_threshold
                            best_match_score = -1

                            for pred_idx in range(len(cls_pred_boxes)):
                                # Skip if prediction is already matched to another ground truth
                                if any(pred_idx == match[1] for match in matched_indices):
                                    continue

                                iou = iou_matrix[gt_idx, pred_idx]
                                score = cls_pred_scores[pred_idx]
                                
                                if iou >= best_match_iou:
                                    best_match_iou = iou
                                    best_match_idx = pred_idx
                                    best_match_score = score

                            if best_match_idx >= 0:
                                matched_indices.append((gt_idx, best_match_idx))
                                matched_gt_indices[cls_id].add(cls_gt_indices[gt_idx])
                                
                                if cls_id == 1:  # Ship
                                    image_detected_ship = True
                                    img_true_positives += 1
                                    
                                # Store best detection for this ground truth
                                gt_global_idx = cls_gt_indices[gt_idx]
                                pred_global_idx = cls_pred_indices[best_match_idx]
                                best_detections[(gt_global_idx, cls_id)] = (pred_global_idx, best_match_score)

                    # Calculate metrics
                    true_positives = len(matched_indices)
                    false_positives = len(cls_pred_boxes) - true_positives
                    false_negatives = len(cls_gt_boxes) - true_positives

                    # Update class metrics
                    class_metrics[cls_id]['true_positives'] += true_positives
                    class_metrics[cls_id]['false_positives'] += false_positives
                    class_metrics[cls_id]['false_negatives'] += false_negatives

                    # FOR SHIP CLASS: Update individual ship detection count
                    if cls_id == 1:  # Ship
                        ship_detection_metrics['correctly_detected_ships'] += true_positives

                    # Store data for mAP calculation
                    for pred_idx in range(len(cls_pred_boxes)):
                        is_true_positive = any(match[1] == pred_idx for match in matched_indices)
                        class_predictions[cls_id].append(1 if is_true_positive else 0)
                        class_scores[cls_id].append(cls_pred_scores[pred_idx])
                
                # Update all_gt_labels and all_pred_labels for confusion matrix
                for label in gt_labels:
                    all_gt_labels.append(label)
                
                for label in pred_labels:
                    all_pred_labels.append(label)

                # If image contains ships and at least one ship was detected, increment counter
                if image_has_ships and image_detected_ship:
                    ship_detection_metrics['correct_ship_images'] += 1

                # Calculate overall metrics for this image
                total_true_positives = sum(class_metrics[cls_id]['true_positives'] for cls_id in [1, 2])
                total_false_positives = sum(class_metrics[cls_id]['false_positives'] for cls_id in [1, 2])
                total_false_negatives = sum(class_metrics[cls_id]['false_negatives'] for cls_id in [1, 2])

                total_precision = total_true_positives / (total_true_positives + total_false_positives) if (
                                                                                                                   total_true_positives + total_false_positives) > 0 else 0
                total_recall = total_true_positives / (total_true_positives + total_false_negatives) if (
                                                                                                                total_true_positives + total_false_negatives) > 0 else 0
                total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (
                                                                                                            total_precision + total_recall) > 0 else 0

                # Ship detection success
                ship_found_percent = 100 if image_has_ships and image_detected_ship else 0

                # Save per-image metrics
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
                    'ship_found': ship_found_percent,  # Ship detection success
                    'inference_time_ms': inference_time * 1000 / len(images)
                })

                # Create figure for visualization (2 subplots: Ground Truth and Predictions)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

                # Left figure: Ground Truth
                ax1.imshow(original_img)
                ax1.set_title("Ground Truth", fontsize=14)
                
                # Ground truth boxes (green for all classes)
                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Use green for ground truth (all classes)
                    color = gt_color if label == 1 else land_color
                    class_name = class_names.get(label, 'Unknown')

                    rect = patches.Rectangle((x1, y1), width, height,
                                             linewidth=2, edgecolor=color,
                                             facecolor='none')
                    ax1.add_patch(rect)
                    ax1.text(x1, y1 - 5, f"{class_name}", color=color, fontsize=10)

                    # Show center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    ax1.plot(center_x, center_y, 'o', color=color, markersize=5)

                ax1.axis('off')

                # Right figure: Predictions
                ax2.imshow(original_img)
                title_suffix = "Center Distance" if use_distance_metric else "IoU"
                ax2.set_title(f"Model Prediction ({title_suffix})", fontsize=14)

                # For visualization, only show the highest scoring detection for each ground truth
                visualized_predictions = set()
                
                # First, draw matched predictions (best detections)
                for (gt_idx, cls_id), (pred_idx, score) in best_detections.items():
                    box = pred_boxes[pred_idx]
                    label = pred_labels[pred_idx]
                    
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Use red for predictions (all classes)
                    color = pred_color if label == 1 else land_color
                    class_name = class_names.get(label, 'Unknown')

                    rect = patches.Rectangle((x1, y1), width, height,
                                             linewidth=2, edgecolor=color,
                                             facecolor='none')
                    ax2.add_patch(rect)
                    ax2.text(x1, y1 - 10, f"{class_name}: {score:.2f}", color=color, fontsize=10)

                    # Show center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    ax2.plot(center_x, center_y, 'o', color=color, markersize=5)
                    
                    visualized_predictions.add(pred_idx)
                
                # Then draw false positives (predictions that weren't matched to any ground truth)
                for pred_idx, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                    if pred_idx in visualized_predictions:
                        continue  # Skip if already visualized
                        
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Use red for predictions (all classes)
                    color = pred_color if label == 1 else land_color
                    class_name = class_names.get(label, 'Unknown')

                    rect = patches.Rectangle((x1, y1), width, height,
                                             linewidth=2, edgecolor=color,
                                             facecolor='none', linestyle='--')  # Dashed line for false positives
                    ax2.add_patch(rect)
                    ax2.text(x1, y1 - 10, f"{class_name}: {score:.2f}", color=color, fontsize=10)

                ax2.axis('off')

                # Show center distance lines (only for ships)
                if use_distance_metric and image_has_ships:
                    for (gt_idx, cls_id), (pred_idx, score) in best_detections.items():
                        if cls_id == 1:  # Ship class
                            gt_box = gt_boxes[gt_idx]
                            pred_box = pred_boxes[pred_idx]
                            
                            distance = calculate_center_distance(gt_box, pred_box)
                            
                            # If distance is below threshold, connect centers with a line
                            if distance < distance_threshold:
                                gt_center_x = (gt_box[0] + gt_box[2]) / 2
                                gt_center_y = (gt_box[1] + gt_box[3]) / 2
                                pred_center_x = (pred_box[0] + pred_box[2]) / 2
                                pred_center_y = (pred_box[1] + pred_box[3]) / 2

                                ax2.plot([gt_center_x, pred_center_x], [gt_center_y, pred_center_y],
                                         color='green', linestyle='--', linewidth=1, alpha=0.7)
                                ax2.text((gt_center_x + pred_center_x) / 2, (gt_center_y + pred_center_y) / 2,
                                         f"{distance:.1f}px", fontsize=7, color='green')

                # Green frame if ships found, red if not found
                ship_detection_status = "Found ✓" if image_has_ships and image_detected_ship else "Not Found ✗"
                status_color = 'green' if image_has_ships and image_detected_ship else 'red'

                # Add overall title
                plt.suptitle(
                    f"Image ID: {image_id} - Ship: {ship_detection_status}",
                    fontsize=14, color=status_color)

                # Save figure
                output_path = os.path.join(output_dir, f"result_id{image_id}.png")
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

    # Calculate mAP for each class
    class_aps = {}

    for cls_id in [1, 2]:  # Ship and Land
        if len(class_scores[cls_id]) == 0:
            class_aps[cls_id] = 0.0
            continue

        # Sort predictions by scores
        sorted_indices = np.argsort(class_scores[cls_id])[::-1]  # Sort in descending order
        class_predictions[cls_id] = np.array(class_predictions[cls_id])[sorted_indices]

        # Calculate Precision and Recall
        cumulative_true_positives = np.cumsum(class_predictions[cls_id])
        cumulative_false_positives = np.cumsum(1 - class_predictions[cls_id])

        precision_curve = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
        recall_curve = cumulative_true_positives / class_metrics[cls_id]['total_gt'] if class_metrics[cls_id][
                                                                                                'total_gt'] > 0 else np.zeros_like(
            cumulative_true_positives)

        # Calculate AP (Average Precision)
        class_aps[cls_id] = calculate_ap(precision_curve, recall_curve)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f'Precision-Recall curve: AP={class_aps[cls_id]:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {class_names[cls_id]}')
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, f'precision_recall_{class_names[cls_id]}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Calculate overall mAP (mean Average Precision)
    mAP = np.mean([ap for ap in class_aps.values()])

    # Calculate class-based metrics
    class_performance = []

    for cls_id in [1, 2]:  # Ship and Land
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
        
    # Calculate and plot confusion matrix
    if len(all_gt_labels) > 0 and len(all_pred_labels) > 0:
        # Calculate confusion matrix using only classes 1 and 2 (Ship and Land)
        filtered_gt = [label for label in all_gt_labels if label in [1, 2]]
        filtered_pred = [label for label in all_pred_labels if label in [1, 2]]
        
        if filtered_gt and filtered_pred:
            cm = confusion_matrix(filtered_gt, filtered_pred, labels=[1, 2])
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[class_names[1], class_names[2]])
            disp.plot(cmap=plt.cm.Blues, values_format='d')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    # Plot per-class metrics as bar charts
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'AP']
    metrics_values = {
        class_names[1]: [class_performance[0]['precision'], class_performance[0]['recall'], 
                       class_performance[0]['f1_score'], class_performance[0]['ap']],
        class_names[2]: [class_performance[1]['precision'], class_performance[1]['recall'], 
                       class_performance[1]['f1_score'], class_performance[1]['ap']]
    }
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, metrics_values[class_names[1]], width, label=class_names[1], color='red')
    rects2 = ax.bar(x + width/2, metrics_values[class_names[2]], width, label=class_names[2], color='blue')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Update ship detection rates
    if ship_detection_metrics['total_ships'] > 0:
        ship_detection_metrics['ship_detection_rate'] = ship_detection_metrics['correctly_detected_ships'] / \
                                                        ship_detection_metrics['total_ships']

    # Image-based detection rate (old detection_rate)
    if ship_detection_metrics['images_with_ships'] > 0:
        ship_detection_metrics['image_detection_rate'] = ship_detection_metrics['correct_ship_images'] / \
                                                         ship_detection_metrics['images_with_ships']
    
    # Plot ship detection rates as a bar chart
    plt.figure(figsize=(10, 6))
    rates = [ship_detection_metrics['ship_detection_rate'] * 100, ship_detection_metrics['image_detection_rate'] * 100]
    labels = ['Individual Ship Detection Rate', 'Image-based Detection Rate']
    
    bars = plt.bar(labels, rates, color=['red', 'blue'])
    plt.ylim(0, 100)
    plt.ylabel('Detection Rate (%)')
    plt.title('Ship Detection Performance')
    
    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ship_detection_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Print performance statistics
    if num_processed > 0:
        avg_time = total_time / num_processed
        fps = 1.0 / avg_time

        print(f"\n--- Performance Statistics ---")
        print(f"Total processed images: {num_processed}")
        print(f"mAP@{iou_threshold}: {mAP:.4f}")
        print(f"Average processing time: {avg_time * 1000:.1f} ms/image")
        print(f"FPS: {fps:.2f}")

        print("\n--- Class-Based Performance ---")
        for cls_perf in class_performance:
            cls_name = cls_perf['class_name']
            print(f"\nClass: {cls_name}")
            print(f"  Ground Truth: {cls_perf['total_gt']}")
            print(f"  Predictions: {cls_perf['total_pred']}")
            print(f"  Precision: {cls_perf['precision']:.4f}")
            print(f"  Recall: {cls_perf['recall']:.4f}")
            print(f"  F1-Score: {cls_perf['f1_score']:.4f}")
            print(f"  AP: {cls_perf['ap']:.4f}")

        # Ship detection metrics - UPDATED
        print("\n--- Ship Detection Statistics ---")
        print(f"Total ships: {ship_detection_metrics['total_ships']}")
        print(f"Correctly detected ships: {ship_detection_metrics['correctly_detected_ships']}")
        print(f"INDIVIDUAL SHIP DETECTION RATE: {ship_detection_metrics['ship_detection_rate'] * 100:.2f}%")
        print(f"Images with ships: {ship_detection_metrics['images_with_ships']}")
        print(f"Images with at least one ship correctly detected: {ship_detection_metrics['correct_ship_images']}")
        print(f"IMAGE-BASED DETECTION RATE: {ship_detection_metrics['image_detection_rate'] * 100:.2f}%")

        # Save to Excel
        # 1. General performance summary
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

        # 2. Class-based performance
        class_performance_df = pd.DataFrame(class_performance)

        # 3. Per-image metrics
        image_metrics_df = pd.DataFrame(image_metrics)

        # 4. Ship detection metrics - UPDATED
        ship_metrics_df = pd.DataFrame([ship_detection_metrics])

        # Save to Excel
        with pd.ExcelWriter(os.path.join(output_dir, 'metric_results.xlsx')) as writer:
            performance_df.to_excel(writer, sheet_name='Overall Performance', index=False)
            class_performance_df.to_excel(writer, sheet_name='Class Performance', index=False)
            image_metrics_df.to_excel(writer, sheet_name='Image Metrics', index=False)
            ship_metrics_df.to_excel(writer, sheet_name='Ship Detection Rate', index=False)

        print(f"Metrics saved to Excel file: {os.path.join(output_dir, 'metric_results.xlsx')}")
        
        # Create a summary text file with key metrics
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write("=== SHIP DETECTION MODEL EVALUATION SUMMARY ===\n\n")
            f.write(f"Model: Faster R-CNN with BiFPN and CBAM\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== OVERALL PERFORMANCE ===\n")
            f.write(f"Total images processed: {num_processed}\n")
            f.write(f"Mean Average Precision (mAP): {mAP:.4f}\n")
            f.write(f"Average processing time: {avg_time * 1000:.1f} ms/image\n")
            f.write(f"Frames per second: {fps:.2f}\n\n")
            
            f.write("=== SHIP CLASS PERFORMANCE ===\n")
            f.write(f"Ship Average Precision: {class_aps[1]:.4f}\n")
            f.write(f"Ship Precision: {class_performance[0]['precision']:.4f}\n")
            f.write(f"Ship Recall: {class_performance[0]['recall']:.4f}\n")
            f.write(f"Ship F1-Score: {class_performance[0]['f1_score']:.4f}\n\n")
            
            f.write("=== SHIP DETECTION RATES ===\n")
            f.write(f"Individual ship detection rate: {ship_detection_metrics['ship_detection_rate'] * 100:.2f}%\n")
            f.write(f"Image-based ship detection rate: {ship_detection_metrics['image_detection_rate'] * 100:.2f}%\n\n")
            
            f.write("=== DETECTION COUNTS ===\n")
            f.write(f"Total ground truth ships: {ship_detection_metrics['total_ships']}\n")
            f.write(f"Correctly detected ships: {ship_detection_metrics['correctly_detected_ships']}\n")
            f.write(f"Images with ships: {ship_detection_metrics['images_with_ships']}\n")
            f.write(f"Images with at least one ship correctly detected: {ship_detection_metrics['correct_ship_images']}\n")
            
        print(f"Summary saved to {os.path.join(output_dir, 'summary.txt')}")


# Ana fonksiyon
def main():
    # Excel dosya yolu
    excel_path = r"C:\gemiv1\data\VeriDetay_v3.xlsx"

    # Create output directory based on script name and current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(__file__).split('.')[0]
    output_dir = os.path.join(r"C:\gemiv1", f"{script_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")

    # Veriyi yükle
    df = load_data_from_excel(excel_path)

    if df is None:
        print("Data could not be loaded, terminating program.")
        return

    print(f"Data successfully loaded: {len(df)} rows.")

    # Sınıf dağılımını kontrol et
    print("\nClass distribution:")
    print(df['class_name'].value_counts())

    # Benzersiz görüntü ID'lerini al
    unique_image_ids = df['idx_image'].unique()

    # Veriyi train ve validation olarak böl
    train_ids, val_ids = train_test_split(unique_image_ids, test_size=0.2, random_state=42)

    print(f"Total {len(unique_image_ids)} images: {len(train_ids)} training, {len(val_ids)} validation")

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
        print("Model loaded: checkpoint['model_state_dict']")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded: checkpoint['state_dict']")
    else:
        try:
            # Belki doğrudan model state dict'tir
            model.load_state_dict(checkpoint)
            print("Model loaded directly as state_dict")
        except:
            print("WARNING: Model could not be loaded! Checkpoint format is not as expected.")
            print("Checkpoint keys:", checkpoint.keys())
            # Basit bir modelle devam et
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

    model.to(device)

    # Modeli değerlendir
    print("Evaluating model...")
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

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()