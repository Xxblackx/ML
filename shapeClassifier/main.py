import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# --- 1. НОВЫЙ, УЛУЧШЕННЫЙ ГЕНЕРАТОР НАБОРА ДАННЫХ ---

def create_diverse_shape_image(shape_type, size=128):
    """
    Создает одно изображение фигуры с вариациями: заливка/контур, толщина, поворот.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    
    center_x = size // 2 + random.randint(-size // 9, size // 9)
    center_y = size // 2 + random.randint(-size // 9, size // 9)
    center = (center_x, center_y)
    
    shape_size = random.randint(size // 5, size // 3)
    angle = random.randint(0, 360)
    color = 255
    
    # С вероятностью 50/50 фигура будет либо залитой, либо контурной
    if random.random() > 0.5:
        thickness = -1  # Залитая фигура
    else:
        thickness = random.randint(2, 6) # Контур со случайной толщиной

    if shape_type == 'circle':
        cv2.circle(img, center, shape_size, color, thickness)
        
    elif shape_type == 'square':
        half = shape_size
        pts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) + center
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_pts = cv2.transform(np.array([pts]), rot_mat)[0].astype(np.int32)
        cv2.drawContours(img, [rotated_pts], 0, color, thickness)

    elif shape_type == 'triangle':
        height = shape_size
        p1 = (center[0], center[1] - height)
        p2 = (center[0] - int(height * 0.866), center[1] + int(height * 0.5))
        p3 = (center[0] + int(height * 0.866), center[1] + int(height * 0.5))
        pts = np.array([p1, p2, p3])
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_pts = cv2.transform(np.array([pts]), rot_mat)[0].astype(np.int32)
        cv2.drawContours(img, [rotated_pts], 0, color, thickness)
    
    noise = np.random.normal(0, 10, (size, size)).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def generate_dataset_on_disk(output_dir='shape_dataset', samples_per_class=6000):
    """
    Создает и сохраняет на диск весь набор данных.
    """
    if os.path.exists(output_dir):
        print(f"Папка '{output_dir}' уже существует. Генерация пропущена.")
        return
        
    print(f"Генерация набора данных в папку '{output_dir}'...")
    os.makedirs(output_dir)
    classes = ['circle', 'square', 'triangle']

    for shape_class in classes:
        class_dir = os.path.join(output_dir, shape_class)
        os.makedirs(class_dir)
            
        print(f"\nСоздание класса: {shape_class}")
        for i in tqdm(range(samples_per_class)):
            img = create_diverse_shape_image(shape_class, size=128)
            filename = os.path.join(class_dir, f"{shape_class}_{i+1}.png")
            cv2.imwrite(filename, img)

    print("\nНабор данных успешно создан!")


# --- 2. АРХИТЕКТУРА МОДЕЛИ И ПАРАМЕТРЫ (БЕЗ ИЗМЕНЕНИЙ) ---

class ShapeClassifier(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        final_conv_channels = 64
        final_size = input_size // 4
        self.fc_input_size = final_conv_channels * final_size * final_size
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Параметры
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.0001


# --- 3. ОСНОВНОЙ СКРИПТ ОБУЧЕНИЯ ---

if __name__ == '__main__':
    # Шаг 1: Сгенерировать набор данных, если его нет
    DATASET_PATH = 'shape_dataset'
    generate_dataset_on_disk(output_dir=DATASET_PATH, samples_per_class=6000)

    # Шаг 2: Определить трансформации и загрузить данные с помощью ImageFolder
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Убедиться, что канал один
        transforms.ToTensor() # Преобразовать в тензор и масштабировать в [0.0, 1.0]
    ])
    
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=data_transforms)
    
    # Шаг 3: Разделить на обучающую и валидационную выборки
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    print(f"\nРазмер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")
    
    # Шаг 4: Инициализация и цикл обучения (без изменений)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = ShapeClassifier(input_size=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Визуализация обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Сохранение модели
    torch.save(model.state_dict(), 'shape_classifier_robust.pth')
    print("Обученная робастная модель сохранена в 'shape_classifier_robust.pth'")