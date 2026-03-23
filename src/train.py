import torch
import logging
import time
from torch.utils.data import DataLoader

from dataset import InhalerDataset
from asformer_model import ASFormer

# Hyperparametry
EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0005
MAX_LEN = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
dataset = InhalerDataset("../data/features_enhanced", "../data/labels", max_len=MAX_LEN)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ROZDIL OPROTI STARE VERZI:
# nova ASFormer konfigurace se self-attention (d_model, num_heads).
model = ASFormer(
    num_layers=8,
    d_model=128,
    input_dim=243,
    num_classes=6,
    num_heads=8,
    dropout=0.1,
    max_dilation=16,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ROZDIL OPROTI STARE VERZI:
# ignorujeme padding labely (-100), pokud mas dataset upraveny na -100 padding.
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Start trenovani na %d videich...", len(dataset))
logger.info(
    "Config: epochs=%d, batch_size=%d, lr=%s, max_len=%d, device=%s",
    EPOCHS,
    BATCH_SIZE,
    LR,
    MAX_LEN,
    device,
)

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0.0

    for step, (data, target, lengths) in enumerate(train_loader, start=1):
        data = data.to(device)       # (B, 243, T)
        target = target.to(device)   # (B, T)

        # ROZDIL OPROTI STARE VERZI:
        # maska validnich pozic pro self-attention.
        valid_mask = target != -100  # (B, T), True = valid, False = padding

        optimizer.zero_grad()

        # ROZDIL OPROTI STARE VERZI:
        # model dostava masku do self-attention.
        output = model(data, mask=valid_mask)  # (B, 6, T)

        loss = criterion(output, target)
        loss.backward()

        # Doporucene pro stabilitu attention modelu.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

        if step % 5 == 0 or step == len(train_loader):
            logger.info(
                "Epoch %d/%d | Step %d/%d | Batch loss: %.4f",
                epoch + 1,
                EPOCHS,
                step,
                len(train_loader),
                loss.item(),
            )

    epoch_time = time.time() - epoch_start
    logger.info(
        "Epoch %d/%d done | Avg loss: %.4f | Time: %.1fs",
        epoch + 1,
        EPOCHS,
        epoch_loss / len(train_loader),
        epoch_time,
    )

# ROZDIL OPROTI STARE VERZI:
# ukladej pod novym jmenem, protoze architektura je jina nez puvodni model.
torch.save(model.state_dict(), "asformer_attention_v1.pth")
logger.info("Model ulozen jako asformer_attention_v1.pth")