import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Pro legendu
from asformer_model import ASFormer
from tkinter import Tk, filedialog
import os

# --- KONFIGURACE (Musí odpovídat train.py) ---
num_layers = 10
num_f_maps = 64
input_dim = 243
num_classes = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEFINICE FÁZÍ A BAREV ---
PHASES_INFO = {
    0: {"name": "KLID", "color": "lightgray"},
    1: {"name": "PRIPRAVA", "color": "royalblue"},
    2: {"name": "ROZDEJCHANI", "color": "orange"},
    3: {"name": "INHALACE", "color": "green"},
    4: {"name": "ZADRZENI", "color": "red"},
    5: {"name": "VYDECH", "color": "purple"}
}

def predict_and_plot_human_readable():
    # 1. Výběr souboru (.npy)
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    feat_path = filedialog.askopenfilename(title="Vyber .npy soubor pro predikci", 
                                          filetypes=[("NumPy files", "*.npy")])
    root.destroy()
    if not feat_path: return

    # 2. Načtení modelu
    model = ASFormer(num_layers, num_f_maps, input_dim, num_classes).to(device)
    model.load_state_dict(torch.load("asformer_v1.pth", map_location=device))
    model.eval()

    # 3. Načtení dat a predikce
    features = np.load(feat_path).T 
    feat_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(feat_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 4. Načtení Ground Truth (anotace .txt)
    gt_path = feat_path.replace("features_enhanced", "labels").replace(".npy", ".txt")
    has_gt = os.path.exists(gt_path)
    gt = np.loadtxt(gt_path, dtype=int) if has_gt else None

    # 5. VIZUALIZACE
    if has_gt:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(15, 3))
        ax1 = None

    time_axis = np.arange(len(prediction))

    # HORNÍ GRAF: GROUND TRUTH
    if ax1:
        gt_colors = [PHASES_INFO[g]["color"] for g in gt]
        ax1.bar(time_axis, [1]*len(gt), color=gt_colors, width=1.0)
        ax1.set_title(f"Anotace (Ground Truth)", fontsize=12, fontweight='bold')
        ax1.set_yticks([]); ax1.set_ylabel(""); ax1.grid(False)

    # DOLNÍ GRAF: PREDIKCE
    pred_colors = [PHASES_INFO[p]["color"] for p in prediction]
    ax2.bar(time_axis, [1]*len(prediction), color=pred_colors, width=1.0)
    ax2.set_title(f"Predikce Modelu (ASFormer)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Snímek (Čas)", fontsize=11)
    ax2.set_yticks([]); ax2.set_ylabel(""); ax2.grid(False)

    # LEGENDA
    legend_patches = []
    for phase, info in PHASES_INFO.items():
        patch = mpatches.Patch(color=info["color"], label=f"{phase}: {info['name']}")
        legend_patches.append(patch)
    
    fig.legend(handles=legend_patches, loc='lower center', ncol=6, 
               fontsize=10, bbox_to_anchor=(0.5, 0.02), shadow=True)

    plt.suptitle(f"Analýza videa: {os.path.basename(feat_path)}", fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.show()

if __name__ == "__main__":
    predict_and_plot_human_readable()