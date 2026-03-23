import torch
from torch.utils.data import DataLoader
from dataset import InhalerDataset
from asformer_model import ASFormer

# Hyperparametry
EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0005
MAX_LEN = 1500 # Přizpůsob podle tvých nejdelších videí

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializace
dataset = InhalerDataset("../data/features_enhanced", "../data/labels", max_len=MAX_LEN)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 243 vstupních features, 6 výstupních tříd (0-5)
model = ASFormer(num_layers=10, num_f_maps=64, input_dim=243, num_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

print(f"Start trénování na {len(dataset)} videích...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for data, target, lengths in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # CrossEntropyLoss očekává (N, C, T)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "asformer_v1.pth")
print("Model uložen jako asformer_v1.pth")