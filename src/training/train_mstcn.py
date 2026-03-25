import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_io.dataset import InhalerDataset
from models.mstcn import MSTCN
from utils.paths import project_paths


EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0005
MAX_LEN = 1000

NUM_STAGES = 4
NUM_LAYERS = 8
NUM_F_MAPS = 64
INPUT_DIM = 243
NUM_CLASSES = 6
DROPOUT = 0.3

LAMBDA_TMSE = 0.15
TAU = 4.0


def lengths_to_mask(lengths, max_len, device):
    lengths = lengths.to(device)
    rng = torch.arange(max_len, device=device).unsqueeze(0)
    return rng < lengths.unsqueeze(1)


def temporal_mse_loss(logits, valid_mask, tau=4.0):
    log_probs = F.log_softmax(logits, dim=1)
    diff = (log_probs[:, :, 1:] - log_probs[:, :, :-1]) ** 2
    diff = torch.clamp(diff, max=tau ** 2)
    diff = diff.mean(dim=1)

    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    pair_mask_f = pair_mask.float()

    denom = pair_mask_f.sum()
    if denom.item() == 0:
        return logits.new_tensor(0.0)

    return (diff * pair_mask_f).sum() / denom


def main():
    paths = project_paths(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = InhalerDataset(
        str(paths["features_enhanced"]),
        str(paths["labels"]),
        max_len=MAX_LEN,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MSTCN(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_f_maps=NUM_F_MAPS,
        dim_in=INPUT_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Start MS-TCN trenovani na {len(dataset)} videich...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_tmse = 0.0

        for data, target, lengths in train_loader:
            data = data.to(device)
            target = target.to(device)

            _, t_steps = target.shape
            valid_mask = lengths_to_mask(lengths, t_steps, device=device)

            target_masked = target.clone()
            target_masked[~valid_mask] = -100

            optimizer.zero_grad()
            stage_outputs = model(data, mask=valid_mask)

            ce_loss_total = 0.0
            tmse_loss_total = 0.0

            for s in range(stage_outputs.size(0)):
                logits = stage_outputs[s]
                ce_loss = criterion_ce(logits, target_masked)
                tmse_loss = temporal_mse_loss(logits, valid_mask, tau=TAU)
                ce_loss_total = ce_loss_total + ce_loss
                tmse_loss_total = tmse_loss_total + tmse_loss

            loss = ce_loss_total + LAMBDA_TMSE * tmse_loss_total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ce += ce_loss_total.item()
            epoch_tmse += tmse_loss_total.item()

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Total: {epoch_loss / len(train_loader):.4f} | "
            f"CE(sum stages): {epoch_ce / len(train_loader):.4f} | "
            f"TMSE(sum stages): {epoch_tmse / len(train_loader):.4f}"
        )

    torch.save(model.state_dict(), "mstcn_v1.pth")
    print("Model ulozen jako mstcn_v1.pth")


if __name__ == "__main__":
    main()
