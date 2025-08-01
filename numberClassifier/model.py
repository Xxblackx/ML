from preprocessing import test_data, train_data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as pyplot
import numpy as np

BATCH_SISE = 64

train_dl = DataLoader(train_data, batch_size=BATCH_SISE, shuffle=True)
valid_dl = DataLoader(test_data, batch_size=BATCH_SISE * 2)


class MnistCNN_stupid(nn.Module):
    def __init__(self):
        super().__init__()

        # kernel - ядро, которое бежит по картинке и 'собирает' признаки
        # stride - шаг, с которым ядро шагает по матрице
        # padding - controls the amount of padding applied to the input. 
        # It can be either a string {‘valid’, ‘same’} or an int / a tuple of ints giving the amount of implicit padding applied on both sides.

        # conv2d - Applies a 2D convolution over an input signal composed of several input planes.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
    
    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28) # как numpy reshape
        xb = F.relu(self.conv1(xb)) # активация после свертки
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1)) # аналог reshape, -1 используется для автоматического расчета формы, xb.size(1) в нашем случае вернет 10 - кол-во классов 
        # -> тензор 64 на 10


# можно сделать светрочные слои в self.features = nn.seq ...
# и класс для самой сети, а в forward  вызвать features, flatten (для "сплющивания в 1 вектор"), model (1 слой = кол-ву выходных нейронов сети features)
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(784,10)
        )
    
    def forward(self, x):
        x =  x.view(-1, 784)
        return self.lin(x)

    def fit(self, epochs, opt, criterion):   
        for epoch in range(epochs):
            model.train() 
            epoch_train_loss = 0.0
            for batch_x, batch_y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                opt.zero_grad() # обнуляем градиенты после предыдущих операций
                predictions = model(batch_x) 
                loss = criterion(predictions, batch_y)
                loss.backward() # обратное распространение 
                opt.step() # рассчет градиента 
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_dl)
            # history['train_loss'].append(avg_train_loss)

            # --- ПРОВЕРКА НА ВАЛИДАЦИОННЫХ ДАННЫХ ---
            model.eval() # отключение параметров, используемых для обучения (dropout ...)

            epoch_val_loss = 0.0
            all_true_labels = []
            all_predicted_labels = []

            with torch.no_grad():
                for batch_features, batch_labels in valid_dl:
                    
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    predictions = model(batch_features)
                    
                    loss = criterion(predictions, batch_labels)
                    epoch_val_loss += loss.item()
                    
                    predicted_classes = torch.argmax(predictions, dim=1)
                    
                    all_true_labels.extend(batch_labels.cpu().numpy())
                    all_predicted_labels.extend(predicted_classes.cpu().numpy())


            avg_val_loss = epoch_val_loss / len(valid_dl)
            accuracy = accuracy_score(all_true_labels, all_predicted_labels)

            # 6. Записать результаты в историю
            # history['val_loss'].append(avg_val_loss)
            # history['val_accuracy'].append(accuracy)
            print(f"Эпоха {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {accuracy:.4f}")
        
        torch.save(model.state_dict(), 'my_cool_model.pth')


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"using: {device} device")
model = MnistCNN().to(device)
# Momentum is a variation on stochastic gradient descent that takes previous updates into account as well and generally leads to faster training.

lr = 5e-2
# opt = optim.Adam(model.parameters(), lr=lr)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.6)

criterion = nn.CrossEntropyLoss()

model.fit(20, opt, criterion)