import torch 
from torch.utils.data import Dataset
import numpy as np
import os

class InhalerDataset(Dataset):
    def __init__(self, features_dir, labels_dir, max_len=2000):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.max_len = max_len
        self.data_list = self._get_data_list()

    def _get_data_list(self):
        # najde vsechny .npy soubory, ktere maji zaroven i .txt label
        samples = []
        for root, _, files in os.walk(self.features_dir):
            for file in files:
                if file.endswith(".npy"):
                    rel_path = os.path.relpath(os.path.join(root, file), self.features_dir)
                    label_file = os.path.join(self.labels_dir, rel_path.replace(".npy", ".txt"))
                    

                    if os.path.exists(label_file):
                        samples.append((os.path.join(root, file), label_file))

            return samples
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        feature_path, label_path = self.data_list[idx]

        # nacteni dat
        features = np.load(feat_path).T # MS-TCN ocekava (C,T) -> (132, pocet_framu)
        labels = np.loadtxt(label_path, dtype=np.int64)

        # osetreni delky
        T = features.shape[1]
        if T < self.max_len:
            features = features[:, :self.max_len]
            labels = labels[:self.max_len]

        # prevod na pyTorch tenzory
        feat_tensor = torch.from_numpy(features).float()
        label_tensor = torch.from_numpy(labels).long()

        return feat_tensor, label_tensor, T
    
if __name__ == "__main__":
    # testovaci spousteni
    ds = InhalerDataset("../data/features_norm", "../data/labels")
    print(f"Nalezeno {len(ds)} kompletnich dvojic (data + label).")









        
