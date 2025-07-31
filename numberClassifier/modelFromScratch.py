#  -- Импорт архива --
from pathlib import Path
import requests

BATCH_SIZE = 64

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"


PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

# -- Распаковка архива --

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# -- Визуализация --
from matplotlib import pyplot
import numpy as np

# pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")

# pyplot.show()
# print(x_train.shape)


# -- Перевод в тензоры --
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

# -- Создание сети с 0 -- 

import math

weights = torch.randn(784, 10) / math.sqrt(784) # 28**2 = 784 пикселя -> 784 входных нейрона
weights.requires_grad_() # автоматически считает градиент после операций
bias = torch.zeros(10, requires_grad=True) # последний слой



def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# @ - matrix mult
def model(xb):
    return log_softmax(xb @ weights + bias)

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
print(xb.size())
preds = model(xb)  # predictions
preds[0], preds.shape
# print(preds[0], preds.shape) # 64 картинки и 10 вариантов ответа -> тензор 64 на 10

# -- Функция потерь --
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll


# -- Проверака точности --
yb = y_train[0:bs]
# print(loss_func(preds, yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# print(accuracy(preds, yb))


# -- Трейнинг луп -- 
# from IPython.core.debugger import set_trace

# !!! вот тут можно поэксперементировать с обучением модели !!!
# попробуйте поиграться с константами и посмотрите ответы.
lr = 0.5  # learning rate
epochs = 10  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # set_trace() # пошаговый дебагер
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

# -- Проверка ответа --


print("anwer cheking")
idx_to_test = 500
img = x_valid[idx_to_test]
with torch.no_grad(): # без града работает
    ans = model(img)

probabilities = torch.exp(ans)

fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(10, 4))

ax1.imshow(img.reshape((28, 28)), cmap="gray")
ax1.set_title(f"Исходное изображение\nПравильный ответ: {y_valid[idx_to_test]}")
ax1.axis('off') # отключаем оси для изображения

digits = np.arange(10)
ax2.bar(digits, probabilities.numpy())
ax2.set_xticks(digits)
ax2.set_title("Предсказания модели")
ax2.set_xlabel("Цифра")
ax2.set_ylabel("Вероятность")

# Показываем оба графика
pyplot.tight_layout()
pyplot.show()

