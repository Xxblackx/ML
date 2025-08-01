import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import random

from preprocessing import test_data 

# class MnistCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin = nn.Sequential(
#             nn.Linear(784,10)
#         )
    
#     def forward(self, x):
#         x =  x.view(-1, 784)
#         return self.lin(x)

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            

        )

        self.lin2 = nn.Sequential(
                nn.Linear(32 * (28//4)**2  , 256),
                nn.ReLU(),
                nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.lin(x)
        x = torch.flatten(x, 1)
        return self.lin2(x)






MODEL_PATH = 'numberClassifier/' 'my_cool_model.pth' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")

model = MnistCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Ошибка: Файл модели не найден по пути '{MODEL_PATH}'")
    exit()


model.eval()


random_idx = random.randint(2500, len(test_data) - 1)


# Получаем тензор изображения и его правильную метку по индексу
# test_data[idx] возвращает кортеж (изображение, метка)
img_tensor, true_label = test_data[random_idx]

print(f"Проверяем изображение с индексом: {random_idx}")
print(f"Его правильная метка: {true_label}")
print(f"Форма исходного тензора изображения: {img_tensor.shape}")


with torch.no_grad():
    # 1. Добавляем "батчевое" измерение: [1, 28, 28] -> [1, 1, 28, 28]
    # 2. Отправляем тензор на то же устройство, что и модель
    img_for_model = img_tensor.unsqueeze(0).to(device)
    
    
    logits = model(img_for_model) # Форма выхода: [1, 10]
    
    probabilities = F.softmax(logits, dim=1) # Форма: [1, 10]


# "Вытаскиваем" вектор вероятностей из батча ([1, 10] -> [10])
probabilities_np = probabilities.squeeze().cpu().numpy()

# Получаем индекс предсказанного класса (индекс с максимальной вероятностью)
predicted_class = np.argmax(probabilities_np)

print(f"Предсказанный моделью класс: {predicted_class}")
print("Вероятности по классам:")
for i, prob in enumerate(probabilities_np):
    print(f"  Цифра {i}: {prob:.4f} ({prob*100:.2f}%)")





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# .squeeze() убирает измерение канала: [1, 28, 28] -> [28, 28]
ax1.imshow(img_tensor.squeeze(), cmap="gray")
ax1.set_title(f"Исходное изображение\nПравильный ответ: {true_label}", fontsize=14)
# Добавляем подзаголовок с предсказанием модели
fig.suptitle(f'Модель предсказала: {predicted_class}', fontsize=16, color='blue' if predicted_class == true_label else 'red')
ax1.axis('off') 


digits = np.arange(10)
bars = ax2.bar(digits, probabilities_np, color='skyblue')


bars[predicted_class].set_color('royalblue')
if predicted_class != true_label:
    bars[true_label].set_color('salmon')

ax2.set_xticks(digits)
ax2.set_title("Предсказанные вероятности", fontsize=14)
ax2.set_xlabel("Цифра")
ax2.set_ylabel("Вероятность")
ax2.set_ylim(0, 1) 


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()
plt.savefig('numberClassifier/' 'Answer.png')