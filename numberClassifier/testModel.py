import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import random

# --- Шаг 1: Загрузка данных и определение архитектуры модели ---
# Важно: Архитектура модели здесь ДОЛЖНА БЫТЬ ТОЧНО ТАКОЙ ЖЕ, как при обучении!
# Поэтому лучше всего импортировать ее из того же файла.
# Предположим, у нас есть доступ к test_data из вашего файла preprocessing.py
# и к классу модели MnistCNN.

# --- Начало блока, который нужно адаптировать под ваш проект ---
# Предположим, этот код находится в том же проекте, что и model.py
from preprocessing import test_data # Убедитесь, что этот импорт работает
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


# --- Конец блока для адаптации ---


# --- Шаг 2: Настройки и загрузка обученной модели ---

MODEL_PATH = 'my_cool_model.pth' # Путь к вашей сохраненной модели

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Создаем экземпляр модели и загружаем в него обученные веса
model = MnistCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Ошибка: Файл модели не найден по пути '{MODEL_PATH}'")
    print("Пожалуйста, сначала обучите и сохраните модель, запустив скрипт обучения.")
    exit()

# Переводим модель в режим оценки - это ОБЯЗАТЕЛЬНО!
# Отключает dropout, настраивает batchnorm и т.д.
model.eval()

# --- Шаг 3: Выбор случайного изображения и его проверка ---

# Выбираем случайный индекс из тестового датасета
random_idx = random.randint(2500, len(test_data) - 1)
# random_idx = 20

# Получаем тензор изображения и его правильную метку по индексу
# test_data[idx] возвращает кортеж (изображение, метка)
img_tensor, true_label = test_data[random_idx]

print(f"\nПроверяем изображение с индексом: {random_idx}")
print(f"Его правильная метка: {true_label}")
print(f"Форма исходного тензора изображения: {img_tensor.shape}")

# --- Шаг 4: Предсказание ---

# Оборачиваем код в torch.no_grad() для экономии памяти и ускорения
with torch.no_grad():
    # 1. Добавляем "батчевое" измерение: [1, 28, 28] -> [1, 1, 28, 28]
    # 2. Отправляем тензор на то же устройство, что и модель
    img_for_model = img_tensor.unsqueeze(0).to(device)
    
    # 3. Делаем предсказание. Модель вернет "логиты"
    logits = model(img_for_model) # Форма выхода: [1, 10]
    
    # 4. Превращаем логиты в вероятности с помощью Softmax
    probabilities = F.softmax(logits, dim=1) # Форма: [1, 10]

# --- Шаг 5: Подготовка данных для визуализации ---

# "Вытаскиваем" вектор вероятностей из батча ([1, 10] -> [10])
# и переводим его в NumPy массив для Matplotlib
probabilities_np = probabilities.squeeze().cpu().numpy()

# Получаем индекс предсказанного класса (индекс с максимальной вероятностью)
predicted_class = np.argmax(probabilities_np)

print(f"Предсказанный моделью класс: {predicted_class}")
print("Вероятности по классам:")
for i, prob in enumerate(probabilities_np):
    print(f"  Цифра {i}: {prob:.4f} ({prob*100:.2f}%)")


# --- Шаг 6: Визуализация ---

# Создаем фигуру с двумя областями для графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Левый график: Изображение ---
# .squeeze() убирает измерение канала: [1, 28, 28] -> [28, 28]
ax1.imshow(img_tensor.squeeze(), cmap="gray")
ax1.set_title(f"Исходное изображение\nПравильный ответ: {true_label}", fontsize=14)
# Добавляем подзаголовок с предсказанием модели
fig.suptitle(f'Модель предсказала: {predicted_class}', fontsize=16, color='blue' if predicted_class == true_label else 'red')
ax1.axis('off') # отключаем оси для изображения

# --- Правый график: Гистограмма вероятностей ---
digits = np.arange(10)
bars = ax2.bar(digits, probabilities_np, color='skyblue')

# Выделяем столбец с предсказанным ответом другим цветом
bars[predicted_class].set_color('royalblue')
# Если ответ неверный, выделяем правильный столбец красным
if predicted_class != true_label:
    bars[true_label].set_color('salmon')

ax2.set_xticks(digits)
ax2.set_title("Предсказанные вероятности", fontsize=14)
ax2.set_xlabel("Цифра")
ax2.set_ylabel("Вероятность")
ax2.set_ylim(0, 1) # Устанавливаем лимит по Y от 0 до 1

# Показываем оба графика
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Оставляем место для suptitle
plt.show()