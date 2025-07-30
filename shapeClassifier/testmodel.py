import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

# 1. Определение архитектуры модели
# ВАЖНО: Эта архитектура должна в точности соответствовать той, 
# что вы использовали для обучения 'shape_classifier_robust.pth'.
# В данном случае - 2 сверточных блока.
# 
# 2
# class ShapeClassifier(nn.Module):
#     def __init__(self, input_size=128):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)

#         )
        
#         final_conv_channels = 64
#         final_size = input_size // 8
#         self.fc_input_size = final_conv_channels * final_size * final_size 
        
#         self.classifier = nn.Sequential(
#             nn.Linear(self.fc_input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(512, 3)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)



# class ShapeClassifier(nn.Module):
#     def __init__(self, input_size=128):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)

#         )
        
#         final_conv_channels = 128
#         final_size = input_size // 16
#         self.fc_input_size = final_conv_channels * final_size ** 2 
        
#         self.classifier = nn.Sequential(
#             nn.Linear(self.fc_input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.35),
#             nn.Linear(512, 3)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)


class ShapeClassifier(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        final_conv_channels = 64
        final_size = input_size // 8
        self.fc_input_size = final_conv_channels * final_size * final_size

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)






# 2. Загрузка обученной робастной модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShapeClassifier(input_size=128).to(device) 


#5! 16!

try:
    # Загружаем веса именно новой, улучшенной модели
    model.load_state_dict(torch.load('shape_classifier(22).pth', map_location=device))
    # model.load_state_dict(torch.load('best_shape_classifier.pth', map_location=device))
    print("Улучшенная модель 'shape_classifier_robust.pth' успешно загружена.")
except FileNotFoundError:
    print("Ошибка: Файл 'shape_classifier_robust.pth' не найден.")
    print("Убедитесь, что вы запустили скрипт обучения на новом датасете и он сохранился.")
    exit()
    
# Переводим модель в режим оценки
model.eval()
# 3. Функция предобработки рисунков (без изменений, она корректна)
def process_paint_image(image_path):
    """
    Эта функция идеально готовит ваши рисунки: загружает, приводит к нужному
    размеру, инвертирует цвета (фигура становится белой) и форматирует в тензор.
    """
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Ошибка: Не удалось найти файл {image_path}")
        return None, None
        
    img = img.resize((128, 128), Image.LANCZOS)
    img_np = np.array(img)
    # Инвертируем: черный (0) -> белый (255), белый (255) -> черный (0)
    img_np = 255 - img_np 
    # Нормализуем в диапазон [0.0, 1.0]
    img_np = img_np.astype(np.float32) / 255.0
    # Преобразуем в тензор нужной формы [1, 1, 128, 128]
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    return img_tensor, img_np

# 4. Процесс тестирования и визуализации (без изменений)
if __name__ == '__main__':
    # Пути к вашим изображениям
    hand_drawn = {
        'circle': 'circle1.png',
        'square': 'square1.png',
        'triangle': 'triangle1.png'
    }

    class_names = ['circle', 'square', 'triangle']

    plt.figure(figsize=(15, 10))
    plot_position = 1

    for shape_name, img_path in hand_drawn.items():
        img_tensor, img_np_for_plot = process_paint_image(img_path)
        if img_tensor is None:
            continue

        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(output, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        
        plt.subplot(2, 3, plot_position)
        plt.imshow(img_np_for_plot, cmap='gray')
        plt.title(f"Input: {shape_name}\nPredicted: {predicted_class}")
        plt.axis('off')
        
        plt.subplot(2, 3, plot_position + 3)
        plt.bar(class_names, probs.cpu().numpy(), color=['red', 'green', 'blue'])
        plt.title(f"Confidence: {probs[predicted_idx].item():.2%}")
        plt.ylim(0, 1)
        plt.ylabel("Probability")
        plot_position += 1

    plt.tight_layout(pad=3.0)
    plt.suptitle("Тестирование на ручных изображениях (Улучшенная Модель)", fontsize=16)
    plt.savefig('paint_test_results_robust.png')
    plt.show()

    print("\nДетальный отчет по ручным изображениям (Улучшенная Модель):")
    for shape_name, img_path in hand_drawn.items():
        img_tensor, _ = process_paint_image(img_path)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
        print(f"\nИзображение: {shape_name}")
        for j, cls in enumerate(class_names):
            print(f"  {cls}: {probs[j]:.2%}")