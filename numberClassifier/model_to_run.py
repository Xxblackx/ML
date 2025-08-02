# --- ИМПОРТЫ ---
from preprocessing import test_data, train_data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 


BATCH_SISE = 64
EPOCHS = 15
LR = 5e-2 # e-1 = 10^-1
MOMENTUM = 0.7 # для SGD 

train_dl = DataLoader(train_data, batch_size=BATCH_SISE, shuffle=True)
valid_dl = DataLoader(test_data, batch_size=BATCH_SISE * 2)

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
        x = torch.flatten(x, 1) # плющим тензор для линейного слоя 
        return self.lin2(x)
            


def fit(epochs, model, train_loader, val_loader, opt, criterion, device):   
    for epoch in range(epochs):
        model.train() 
        epoch_train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            opt.zero_grad()
            predictions = model(batch_x) 
            loss = criterion(predictions, batch_y)
            loss.backward()
            opt.step()
            
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        all_true_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                predictions = model(batch_x)
                # --- МЕТРИКИ ---
                loss = criterion(predictions, batch_y)
                epoch_val_loss += loss.item()
                predicted_classes = torch.argmax(predictions, dim=1)
                
                all_true_labels.extend(batch_y.cpu().numpy())
                all_predicted_labels.extend(predicted_classes.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {accuracy:.4f}")
        
    print("Обучение завершено.")
    
    # --- МАТРИЦА ОШИБОК ---

    cm = confusion_matrix(all_true_labels, all_predicted_labels)

    cm_errors = cm.copy() 
    # главная диагональ - нули
    np.fill_diagonal(cm_errors, 0)
 
    # --- ОТРИСОВКА МАТРИЦЫ ---   
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_errors, 
        annot=True,     
        fmt='d',        
        cmap='Reds',   
        xticklabels=range(10), 
        yticklabels=range(10)  
    )

    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.title('Матрица ошибок', fontsize=15)

    # 5. Показываем график
    plt.savefig('numberClassifier/' 'Matrix.png')
    plt.show()


    torch.save(model.state_dict(), 'numberClassifier/' 'my_cool_model.pth')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}")
    model = MnistCNN().to(device)

    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    fit(EPOCHS, model, train_dl, valid_dl, opt, criterion, device)